# Data

Large files here are git-ignored. Expected layout:

```
data/
  wordlists/codenames.txt                 # Codenames base-game words, one per line
  embeddings/glove.6B.300d.txt            # GloVe (Pennington et al. 2014)
  embeddings/numberbatch-en-19.08.txt.gz  # ConceptNet Numberbatch (Speer et al. 2017)
  swow/strength.SWOW-EN.R123.csv          # Small World of Words (De Deyne et al. 2019)
```

- **Embeddings:** `bash scripts/download_data.sh`.
- **SWOW:** request/download the English R123 strength file from
  https://smallworldofwords.org/en/project/research and place it in `data/swow/`.
  Column names are configurable in `codenames_rsa/meaning/swow.py:load_swow_strengths`.
- **Codenames wordlist:** add the publicly available base-game wordlist to
  `data/wordlists/codenames.txt` (lower-case, one word per line). Multi-word entries
  such as "ice cream" should be dropped or joined, since the meaning models are
  single-word.
