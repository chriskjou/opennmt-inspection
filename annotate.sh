#!/bin/sh

diff \
--old-line-format='
' \
--new-line-format='
' \
--unchanged-line-format=' %l
' \
"$@"
