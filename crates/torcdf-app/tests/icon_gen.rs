//! One-off converter from the owner's artwork (`icon/icon_v1.png`, not
//! committed) to `assets/torcdf.ico` and `assets/torcdf-256.png`.
//!
//! Run: `ICON_SRC=<path to icon_v1.png> cargo test -p torcdf-app --test icon_gen -- --ignored`

use std::collections::VecDeque;
use std::path::PathBuf;

/// RGBA image, row-major.
struct Img {
    w: usize,
    h: usize,
    px: Vec<[u8; 4]>,
}

fn decode(path: &PathBuf) -> Img {
    let file = std::io::BufReader::new(std::fs::File::open(path).expect("icon source"));
    let mut dec = png::Decoder::new(file);
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().unwrap();
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        c => panic!("unexpected colour type {c:?}"),
    };
    let px = (0..w * h)
        .map(|i| {
            let s = &buf[i * ch..i * ch + ch];
            match ch {
                3 => [s[0], s[1], s[2], 255],
                4 => [s[0], s[1], s[2], s[3]],
                1 => [s[0], s[0], s[0], 255],
                _ => [s[0], s[0], s[0], s[1]],
            }
        })
        .collect();
    Img { w, h, px }
}

/// Near-black pixels joined to the border become transparent.
fn clear_surround(img: &mut Img) {
    let dark = |p: [u8; 4]| p[0].max(p[1]).max(p[2]) < 12;
    let mut seen = vec![false; img.w * img.h];
    let mut q = VecDeque::new();
    for x in 0..img.w {
        q.push_back((x, 0));
        q.push_back((x, img.h - 1));
    }
    for y in 0..img.h {
        q.push_back((0, y));
        q.push_back((img.w - 1, y));
    }
    while let Some((x, y)) = q.pop_front() {
        let i = y * img.w + x;
        if seen[i] || !dark(img.px[i]) {
            continue;
        }
        seen[i] = true;
        img.px[i] = [0, 0, 0, 0];
        if x > 0 {
            q.push_back((x - 1, y));
        }
        if y > 0 {
            q.push_back((x, y - 1));
        }
        if x + 1 < img.w {
            q.push_back((x + 1, y));
        }
        if y + 1 < img.h {
            q.push_back((x, y + 1));
        }
    }
}

/// Square crop around the opaque pixels.
fn crop_square(img: &Img) -> Img {
    let (mut x0, mut y0, mut x1, mut y1) = (img.w, img.h, 0, 0);
    for y in 0..img.h {
        for x in 0..img.w {
            if img.px[y * img.w + x][3] > 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    let side = (x1 - x0 + 1).max(y1 - y0 + 1);
    let (ox, oy) = (x0 as isize - ((side - (x1 - x0 + 1)) / 2) as isize, y0 as isize - ((side - (y1 - y0 + 1)) / 2) as isize);
    let mut px = vec![[0u8; 4]; side * side];
    for y in 0..side {
        for x in 0..side {
            let (sx, sy) = (ox + x as isize, oy + y as isize);
            if sx >= 0 && sy >= 0 && (sx as usize) < img.w && (sy as usize) < img.h {
                px[y * side + x] = img.px[sy as usize * img.w + sx as usize];
            }
        }
    }
    Img { w: side, h: side, px }
}

/// Area-average downscale with premultiplied alpha.
fn shrink(img: &Img, n: usize) -> Img {
    let mut px = Vec::with_capacity(n * n);
    for ty in 0..n {
        let (sy0, sy1) = (ty * img.h / n, ((ty + 1) * img.h / n).max(ty * img.h / n + 1));
        for tx in 0..n {
            let (sx0, sx1) = (tx * img.w / n, ((tx + 1) * img.w / n).max(tx * img.w / n + 1));
            let mut acc = [0f64; 4];
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    let p = img.px[y * img.w + x];
                    let a = p[3] as f64 / 255.0;
                    for c in 0..3 {
                        acc[c] += p[c] as f64 * a;
                    }
                    acc[3] += a;
                }
            }
            let count = ((sy1 - sy0) * (sx1 - sx0)) as f64;
            let a = acc[3] / count;
            let un = |v: f64| if acc[3] > 0.0 { (v / acc[3]).round().clamp(0.0, 255.0) as u8 } else { 0 };
            px.push([un(acc[0]), un(acc[1]), un(acc[2]), (a * 255.0).round() as u8]);
        }
    }
    Img { w: n, h: n, px }
}

fn encode(img: &Img) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut e = png::Encoder::new(&mut out, img.w as u32, img.h as u32);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        let mut w = e.write_header().unwrap();
        let flat: Vec<u8> = img.px.iter().flatten().copied().collect();
        w.write_image_data(&flat).unwrap();
    }
    out
}

/// An .ico whose entries are PNG-compressed.
fn ico(entries: &[(usize, Vec<u8>)]) -> Vec<u8> {
    let mut out = vec![0, 0, 1, 0];
    out.extend((entries.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * entries.len();
    for (size, data) in entries {
        let s = if *size >= 256 { 0 } else { *size as u8 };
        out.extend([s, s, 0, 0]);
        out.extend(1u16.to_le_bytes());
        out.extend(32u16.to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend((offset as u32).to_le_bytes());
        offset += data.len();
    }
    for (_, data) in entries {
        out.extend(data);
    }
    out
}

#[test]
#[ignore]
fn generate_icon() {
    let src = std::env::var_os("ICON_SRC").map(PathBuf::from).unwrap_or_else(|| "../../icon/icon_v1.png".into());
    let mut img = decode(&src);
    clear_surround(&mut img);
    let sq = crop_square(&img);
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let big = shrink(&sq, 256);
    std::fs::write(assets.join("torcdf-256.png"), encode(&big)).unwrap();
    let entries: Vec<(usize, Vec<u8>)> = [256, 64, 48, 32, 24, 16]
        .iter()
        .map(|&n| (n, encode(&shrink(&sq, n))))
        .collect();
    let bytes = ico(&entries);
    assert!(bytes.len() < 200_000, "ico is {} bytes", bytes.len());
    std::fs::write(assets.join("torcdf.ico"), bytes).unwrap();
    assert_eq!(big.px[0][3], 0, "corner transparent");
}
