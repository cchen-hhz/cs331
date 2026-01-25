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