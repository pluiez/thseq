# thseq

基于 PyTorch 的 NMT 工具。

## Features

- Transformer in [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- 分布式训练
- 混合精度
- Batched beam search
- 集成解码
- 转换 [fairseq](https://github.com/pytorch/fairseq) 的标准 Transformer 模型检查点为 thseq 兼容的格式
- Tensorboard logging if Tensorboard is available

## Requirements

+ torch==1.3.0+
+ Lunas=0.4.0+

thseq partially builds upon [Lunas](https://github.com/pluiez/Lunas/),  a stand-alone data processing library offering concise APIs and logics for processing arbitrary input formats (e.g., image and text), highly customizable, easy to use and worth a try!

## Directory Structure

1. thseq is organized as follows:

   ```
   ROOT directory
   - scripts: helper scripts
   - thseq: source code directory
   - generate.py: sequence generation
   - train.py: training
   ```

2. Checkpoints and configuration are saved in a specified CHECKPOINT_DIR and organized as follows:

   ```
   CHECKPOINT_DIR
   - config.json: hyper parameters and runtime options
   - checkpoints.json: metadata for checkpoint files
   - ckp.E{EPOCH}.S{GLOBAL_STEP}.pt: checkpoint file in terms of epochs
   - ckp.S{GLOBAL_STEP}.pt: checkpoint file in terms of steps
   - ckp.T{TIMESTAMP}.S{GLOBAL_STEP}.pt: checkpoint file in terms of seconds
   - tensorboard (optional): directory for TensorBoard loggings
   ```

## WMT'14 En-Fr Translation 实例

Here we take WMT'14 English-French translation as an example to train a Transformer model for demonstration.

+ 数据准备

  - 训练集: ~3.6M sentence pairs

  - 开发/测试集: newstest2013/newstest2014

  - Segment the source and target languages jointly with BPE of 40000 merge operations

  - 生成词表

    ```shell
    DATA=~/data/wmt14/enfr/wmt14_en_fr
    # 分别生成词表
    python scripts/get_vocab.py $DATA/train.en -P 6 > $DATA/vocab.en
    python scripts/get_vocab.py $DATA/train.fr -P 6 > $DATA/vocab.fr
    # 生成合并词表
    python scripts/join_vocab.py $DATA/vocab.?? > $DATA/vocab.join.en
    cp $DATA/vocab.join.en $DATA/vocab.join.fr
    ```
    
  - 预处理:
  
    ```shell
    python scripts/binarize_data.py \
    --train $DATA/train.* \
    --vocab  $DATA/vocab.join.* \
    --vocab-size 0 0 \
    --vocab-min-freq 10 10 \
    --target-dir $DATA/bin # 输出目录
    ```
  
    

- 训练

  - 参考 [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) 的设置训练 Transformer-base 模型：

    ```shell
    python train.py \
    --checkpoint enfr-base \
    --model transformer \
    --langs en fr \ # 语言后缀（可选）
    --train-bin $DATA/bin \
    --train train \ # 自动扩展后缀，如果提供了`--langs` 
    --vocab vocab.join \ # 自动扩展后缀，如果提供了`--langs` 
    --dev $DATA/test13 \ # 自动扩展后缀，如果提供了`--langs` 
    --max-step 100000 \
    --max-epoch 0 \
    --lr 7e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 \
    --warmup-steps 4000 \
    --optimizer adam \
  --num-workers 6 \
    --max-tokens 8333 \ # 有效batch大小为 (max_tokens * dist_world_size * accumulate)
  --dist-world-size 3 \ # gpu 数量，*必须*<=可用GPU数量，否则无法启动分布式
    --accumulate 1 \
  --seed 9527 \ 
    --save-checkpoint-steps 2000 \
    --save-checkpoint-secs 0 \
    --save-checkpoint-epochs 1 \
    --keep-checkpoint-max 10 \
    --keep-best-checkpoint-max 2 \
    --shuffle 1 \
    --input-size 512 \
    --hidden-size 512 \
    --ffn-hidden-size 2048 \
    --num-heads 8 \
    --share-all-embedding 1 \
    --residual-dropout 0.1 \
    --attention-dropout 0 \
    --ffn-dropout 0 \
    --val-method bleu \ # 验证指标: bleu/logp
    --val-steps 1000 \ # 验证频率
    --val-max-tokens 4096 \ # validation batch-size
    --fp16 half \ # 混合精度训练: none/half/amp
    > log.train 2>&1 
    
    
    ```
    
    Type `python train.py -h` for more available options for model, optimizer, lr_scheduler, dataset, etc.

- 解码

  - Run the following to generate translations with a trained model:

    ```shell
    python generate.py $DATA/test14.en \
    -k 4 \ beam size
--alpha 1.0 \ length penalty
    --max-tokens 8192 \ # maximum source tokens per batch
--checkpoints enfr-base/ckp.S00100000.pt \ # pass multiple checkpoints for ensembled decoding
    --num-workers 0 \
    > mt.test14.fr
    ```
    
    Optionally, replace `$DATA/test14.en` with a hyphen `-`  to read from standard input. Type `python generate.py -h` for more options.
    
  
- 评测

  |                                                          | newstest2014 (tok./ detok. BLEU) | wps/sps/wall |
  | -------------------------------------------------------- | -------------------------------- | ------------ |
  | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | 38.1/NaN                         | /            |
  | thseq                                                    | 39.94/36.9                       | /            |

  - wps: decoding speed measured in word per second.

  - sps: decoding speed measured in sentences per second.
  - wall: wall time of generation including data loading and processing.

  


## Todo Features

- Sampling-based generation