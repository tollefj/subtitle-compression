# Translate and Compress
Download data from OpenSub/EuroParl from the `/data` folder. Each folder has separate scripts for this.
- We include a batch downloader for several languages. For specific languages:
    - `python download_langs.py en fr` for english--french parallel data


Fine-tuning of models can be seen in `finetune.py`. 

Example printouts of experiments on both in-domain (opensubtitles) and out-of-domain (europarl) are in their respective jupyter notebooks in the root dir.
