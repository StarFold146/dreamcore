"""dreamcore 主体曝光：主体打成白、边上晕光，背景染色、颗粒。详见 README.md。
python dream.py 输入 输出 [blue|lavender|sakura|ember] [--mode auto|sky|seg|luma] [--true] [--keep 0.03] [--glow 1.8] [--band 0.12]"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from rembg import remove, new_session

TONES = {"blue": (92, 112, 168), "lavender": (150, 140, 200), "sakura": (214, 160, 190), "ember": (190, 110, 80)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, dst = args[0], args[1]
    tone = np.array(TONES.get(args[2] if len(args) > 2 else "lavender"), np.float32) / 255
    keep = float(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else 0.12
    im = Image.open(src).convert("RGB")
    im.thumbnail((2000, 2000))
    W, H = im.size
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "auto"
    a0 = np.asarray(im, np.float32) / 255
    mask = None; skysoft = None
    if mode == "luma":
        # 全图处理：不分主体，整张按明暗翻——暗的（树干、枝叶）打白，亮的（天、水光）留成色调
        l = a0 @ np.array([.299, .587, .114], np.float32)
        lo, hi = np.percentile(l, 25), np.percentile(l, 75)
        mask = Image.fromarray((np.clip((hi - l) / max(1e-3, hi - lo), 0, 1) ** .8 * 255).astype(np.uint8))
    elif mode == "seg":
        # 语义分割（SegFormer, ADE20K 150 类）：天、草、水、路、地这些算背景，其余（房子、树、人、花……）算主体；
        # 贴着天的边再用亮度细抠一遍，树梢枝杈才清楚
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        name = "nvidia/segformer-b4-finetuned-ade-512-512"
        proc = SegformerImageProcessor.from_pretrained(name); net = SegformerForSemanticSegmentation.from_pretrained(name).eval()
        with torch.no_grad():
            lg = net(**proc(images=im, return_tensors="pt")).logits
            lg = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
        lab = lg.argmax(1)[0].numpy()
        names = net.config.id2label
        BG = ("sky", "grass", "field", "water", "sea", "river", "lake", "road", "sidewalk", "floor", "earth", "ground", "sand", "path", "dirt", "land", "hill", "mountain", "pool", "waterfall", "fountain", "runway", "snow")
        bgid = [i for i, n in names.items() if any(n.split(";")[0].strip().startswith(b) for b in BG)]
        seg = (~np.isin(lab, bgid)).astype(np.float32)
        l = a0 @ np.array([.299, .587, .114], np.float32)
        skyid = [i for i, n in names.items() if n.startswith("sky")]
        skyref = l[np.isin(lab, skyid)]
        if skyref.size > 500:   # 天上的亮度当基准，比天暗的才算枝叶
            lo, hi = np.percentile(skyref, 5), np.percentile(skyref, 60)
            skyl = np.clip((l - (lo - .08)) / max(1e-3, hi - lo + .08), 0, 1)
            near = np.asarray(Image.fromarray((np.isin(lab, skyid) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15)), np.float32) / 255
            seg = np.where(near > 0, seg * (1 - skyl * near), seg)
            seg = np.maximum(seg, (1 - skyl) * near * np.isin(lab, skyid))
        # 主体里透出来的天和水面倒影（又亮又偏蓝）抠掉，树林里才分得出一根根树干
        b_r = a0[..., 2] - a0[..., 0]
        skyish = np.clip((l - .5) / .22, 0, 1) * np.clip((b_r + .02) * 7, 0, 1)
        seg = seg * (1 - skyish)
        # 零碎小白斑（水面上误判的）去掉：先开运算再合回去
        mb = Image.fromarray((seg > .5).astype(np.uint8) * 255)
        big = mb.filter(ImageFilter.MinFilter(9)).filter(ImageFilter.MaxFilter(13))
        seg = seg * (np.asarray(big, np.float32) / 255)
        mask = Image.fromarray((np.clip(seg, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
        print("subject classes:", sorted({names[i] for i in np.unique(lab) if i not in bgid}))
    elif mode != "sky":
        mask = remove(im, only_mask=True, session=new_session("isnet-general-use"))
        cov = np.asarray(mask, np.float32).mean() / 255
        if mode == "auto" and not (.03 < cov < .9):
            mask = None   # 抠不出像样的主体（风景、天空下的树）就换天空法
    if mask is None:
        # 天空法：亮的、偏蓝/偏灰白的是天，剩下比天暗的枝杈、花、人都算主体
        l = a0 @ np.array([.299, .587, .114], np.float32)
        blue = a0[..., 2] - a0[..., 0]
        sky = np.clip((l - np.percentile(l, 35)) / max(1e-3, np.percentile(l, 85) - np.percentile(l, 35)), 0, 1) * np.clip(.6 + blue * 3, 0, 1)
        sky = np.maximum(sky, np.clip((blue - .08) * 6, 0, 1) * np.clip(l * 3, 0, 1))   # 顶上深一点的蓝天也算天，别被当成主体洗白
        sub = np.clip(1 - sky * 1.4, 0, 1)
        # 只要「顶着天」的那一层：每一列从上往下找最后一块天（只看上面 65%），往下再留一截，
        # 草地、水面这些远离天际线的暗处留在背景里
        isky = sky > .5
        skysoft = sky
        top = isky[: int(H * .65)]
        last = np.where(top.any(0), top.shape[0] - 1 - np.argmax(top[::-1], 0), 0).astype(np.float32)
        k = max(3, W // 60); pad = np.pad(last, k, mode="edge")
        last = np.max(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), 1)
        g = np.exp(-np.linspace(-2, 2, 4 * k + 1) ** 2); g /= g.sum()
        last = np.convolve(np.pad(last, 2 * k, mode="edge"), g, "valid")
        band = float(sys.argv[sys.argv.index("--band") + 1]) if "--band" in sys.argv else .12
        rows = np.arange(H, dtype=np.float32)[:, None]
        reach = np.clip(1 - (rows - last[None, :] - band * H) / (H * .012), 0, 1)
        mask = Image.fromarray((sub * reach * 255).astype(np.uint8))
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.8)), np.float32)[..., None] / 255
    a = np.asarray(im, np.float32) / 255
    lum = (a @ np.array([.299, .587, .114], np.float32))[..., None]
    # 背景：去掉一半饱和、压对比、抬黑位，往色调里染
    # --true：背景尽量留原色（10/17 她说淡淡的也好看，但更想要原来的颜色）
    if "--true" in sys.argv:
        bg = a * .94 + tone * .06
    else:
        bg = lum + (a - lum) * .7
        bg = .5 + (bg - .5) * .92
        bg = bg * .78 + tone * .22
    # 主体：打白，留一点原图明暗当纹理
    sub = 1 - (1 - lum) * keep
    out = bg * (1 - m) + sub * m
    # 光晕：遮罩糊开，往背景上屏幕叠一层白
    glow = np.asarray(mask.filter(ImageFilter.GaussianBlur(max(W, H) * .012)), np.float32)[..., None] / 255
    wide = np.asarray(mask.filter(ImageFilter.GaussianBlur(max(W, H) * .045)), np.float32)[..., None] / 255
    gk = float(sys.argv[sys.argv.index("--glow") + 1]) if "--glow" in sys.argv else 1.0
    halo = np.clip((glow * .45 + wide * .14) * gk, 0, 1) * (1 - m)
    if skysoft is not None: halo = halo * np.clip(skysoft[..., None] * 1.6, 0, 1)   # 光只往天上晕，别糊到草地水面上
    out = 1 - (1 - out) * (1 - halo * np.array([1, .98, 1], np.float32))
    # 颗粒
    rng = np.random.default_rng(7)
    out = out + rng.normal(0, .035, (H, W, 1)).astype(np.float32) + rng.normal(0, .015, (H, W, 3)).astype(np.float32)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst, quality=92)
    print("wrote", dst, W, H)


if __name__ == "__main__":
    main()
