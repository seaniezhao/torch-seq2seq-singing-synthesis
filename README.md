# torch-seq2seq-singing-synthesis

### pytorch implementation of https://mtg.github.io/singing-synthesis-demos/transformer/

##### dataset used：https://wenet.org.cn/opencpop/

to try:
- edit ROOT_PATH in config.py
- create a folder named "raw" in ROOT_PATH and drop wavs and textgrids in pairs
- run preprocess/cut_wav.py
- run prepare_train_data.py
- run trainer.py

todo：
- [ ] tensorboard
- [ ] hifi-gan vocoder
- [ ] pitch model
