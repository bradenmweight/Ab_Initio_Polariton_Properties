#!/bin/bash

# Makes permanent dipoles
$1 << EOF
geometry.fchk
18
5
geometry.out
2
EOF

# Makes transition dipoles
$1 << EOF
geometry.fchk
18
5
geometry.out
4
EOF

