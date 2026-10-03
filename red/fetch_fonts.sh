#!/usr/bin/env bash
# Downloads the open-licensed fonts the renderer uses into ./fonts
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts && cd fonts
N=https://raw.githubusercontent.com/notofonts/noto-cjk/main
G=https://raw.githubusercontent.com/google/fonts/main/ofl
F=https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts
curl -sSL -o NotoSerifSC.otf          "$N/Serif/SubsetOTF/SC/NotoSerifSC-Light.otf"
curl -sSL -o NotoSerifSC-SemiBold.otf "$N/Serif/SubsetOTF/SC/NotoSerifSC-SemiBold.otf"
curl -sSL -o NotoSerifSC-Black.otf    "$N/Serif/SubsetOTF/SC/NotoSerifSC-Black.otf"
curl -sSL -o NotoSansSC-Black.otf     "$N/Sans/SubsetOTF/SC/NotoSansSC-Black.otf"
curl -sSL -o NotoSansKR.otf           "$N/Sans/SubsetOTF/KR/NotoSansKR-Bold.otf"
curl -sSL -o NotoSansDeva.ttf         "$F/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"
curl -sSL -o NotoSansArabic.ttf       "$F/NotoSansArabic/hinted/ttf/NotoSansArabic-Bold.ttf"
curl -sSL -o NotoSansHebrew.ttf       "$F/NotoSansHebrew/hinted/ttf/NotoSansHebrew-Bold.ttf"
curl -sSL -o NotoSansEgypt.ttf        "$F/NotoSansEgyptianHieroglyphs/hinted/ttf/NotoSansEgyptianHieroglyphs-Regular.ttf"
curl -sSL -o Unifraktur.ttf           "$G/unifrakturmaguntia/UnifrakturMaguntia-Book.ttf"
curl -sSL -o Anton.ttf                "$G/anton/Anton-Regular.ttf"
curl -sSL -o Cinzel.ttf               "$G/cinzel/Cinzel%5Bwght%5D.ttf"
curl -sSL -o wenkai.ttf               "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKai-Regular.ttf"
curl -sSL -o jb.zip "https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip"
unzip -o -q -j jb.zip "fonts/ttf/JetBrainsMono-Regular.ttf" && rm jb.zip
