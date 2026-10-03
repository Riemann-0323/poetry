#!/usr/bin/env bash
# Downloads the open-licensed fonts the renderer uses into ./fonts
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts && cd fonts
curl -sSL -o NotoSerifSC.otf      "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/SubsetOTF/SC/NotoSerifSC-Light.otf"
curl -sSL -o NotoSerifSC-Bold.otf "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/SubsetOTF/SC/NotoSerifSC-Medium.otf"
curl -sSL -o wenkai.ttf           "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKai-Regular.ttf"
curl -sSL -o wenkaimono.ttf       "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKaiMono-Regular.ttf"
curl -sSL -o CormorantGaramond-LightItalic.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf"
curl -sSL -o jb.zip "https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip"
unzip -o -q -j jb.zip "fonts/ttf/JetBrainsMono-Light.ttf" && rm jb.zip
