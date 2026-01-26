# 问答栏

用于写简短问题回答。

## BPE

### Understanding Unicode 

- What Unicode character does chr(0) return?

> \x00，空字符

- How does this character’s string representation (__repr__()) differ from its printed representa-tion?

> repr 会返回 '\x00'，print 调用 __str__ 直接输出空白

- What happens when this character occurs in text?

> python cli 中，直接 debug 输出调用 __repr__，print 调用 __str__，会体现区别。

### Unicode Encodings

- What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32?

> 别的编码表示单个字符太长了

- Consider the following (incorrect) function

> 大部分字符由多 bytes 编码，比如 "牛"，错误编码会异常。

- Give a two byte sequence that does not decode to any Unicode character(s).

> 0xFF

### BPE impl

- BPE Training on TinyStories

> 十个进程一起跑的，大致总时长 2min 以内，瓶颈在读入和预分词处理上，性能分析也能看出来。
> 内存峰值大致 11GB（6GB + 5GB swap），期间有将 count 转化为 generator 优化内存。
> 词元上，可以看到类似 Tony, horizon 等单次都能学习到。

-  BPE Training on OpenWebText

> 本地机子没实力跑。

### tokenizer impl

- Implementing the tokenizer

> 其实最后一个测试 XFAIL 是刻意不让通过的，直接加载全部文件还是太吃了。
> 一些踩坑：
> regex.split 不会保留配对 pattern，正则匹配时要用 () 包裹匹配获得匹配组，这样就能保留了。
> 对 special_token 排序是必要的，一个测试测出了这个错误：<|endoftext|><|endoftext|>
> 应该没了，采用的每次循环替换，比较低能，不过单个单次很短，应该没什么问题了，没有必要改链表。

- Experiments with tokenizers

> 这里统一采用 TinyStory 的 10k 词汇表，WebText 的练不起QED
> compare for TinyStory-cut: 12976 and 3118, rate 0.24028976572133168
> compare for owt-cut: 10147 and 3046, rate 0.30018724746230413