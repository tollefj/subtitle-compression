#!/bin/bash
declare -A languages
languages=(
    ["french"]="fr"
    ["polish"]="pl"
    ["romanian"]="ro"
    ["norwegian"]="no"
    ["japanese"]="ja"
    ["korean"]="ko"
)
languages=(
    # ["french"]="fr"
    # ["polish"]="pl"
    # ["hungarian"]="hu"
    # ["norwegian"]="no"
    # ["japanese"]="ja"
    # ["korean"]="ko"
    ["finnish"]="fi"
    ["lithuanian"]="lt"
    ["romanian"]="ro"
)

for lang in "${!languages[@]}"; do
    python3 download_langs.py "en" "${languages[$lang]}"
done