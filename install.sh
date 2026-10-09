export TERM=xterm-256color
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra gpu --no-install-package flash-attn
uv sync --extra gpu