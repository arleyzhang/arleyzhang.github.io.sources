---
title: hexo博客搭建
categories: 随笔
tags:
  - hexo
  - 博客搭建
copyright: true
top: 100
mathjax: true
description: hexo博客搭建。
typora-copy-images-to: ..\pictures\images-hexo_blog
abbrlink: afcbc60d
password:
---



# 1 安装hexo

- 下载node.js并安装（官网下载安装）。
- 下载安装git（官网下载安装），并配置好秘钥。
- 下载安装hexo。打开cmd 运行*npm install -g hexo*。

最好参考官网的安装方法，[点击这里](https://hexo.io/zh-cn/docs/index.html) 。

具体安装方法，这里不再说了。

# 2 建站

hexo 安装好之后要开始准备建立网站：

```shell
$ hexo init <folder>
$ cd <folder>
$ npm install
```

第一句：中文件夹可以是任意的文件夹，你可以专门建立一个文件夹用于存放博客文章，比如 这样 D:\myBlog 。`hexo init` 这个命令是用于初始化你这个文件夹，就相当于 `git init ` 的作用；

第二句：进入 你建的那个文件夹；

第三句：是安装依赖文件。



执行完上述命令后会在你指定的文件夹里生成一系列文件，目录如下：

```
.
├── _config.yml
├── package.json
├── scaffolds
├── source
|   ├── _drafts
|   └── _posts
└── themes
```



完了之后要进入到你建立的这个文件夹，这里就是D:\myBlog文件夹，使用文本编辑器打开 **“_config.yml”** 这个文件，在文件最后面有这两行：

```yaml
deploy:
  type:
```

将这两行改为：

```yaml
deploy:
  type: git
  repo: https://github.com/your_username/your_username.github.io.git
  branch: master
```

注意两点：

1. `type` `repo` `branch` 关键字的冒号后面一定要有一个 **英文空格！！！！** 我第一次就是没有加空格，后面怎么搞都发布不了文章，好郁闷。
2. `repo:` 后面的你的github 仓库地址，根据需要 将 **your_username** 改为你的用户名。
3. `branch:` 这个也可以改为其他分支，这里就不改了。




# 3 使用hexo向github仓库部署

首先安装这个 hexo-deployer-git ，估计就是 hexo 调用 git 的一个包吧，其实hexo 发布文章的根本还是使用的git工具：

```shell
npm install hexo-deployer-git --save
```



接下来准备发布文章，如下：

```shell
$ hexo clean 
$ hexo g
$ hexo d
```

- `hexo clean` 为清除上次生成的内容。

- `hexo g` 根据 博客文章 生成 静态网页文件还有一些格式文件，这部完成后会在 D:\myBlog 中生成 一个 名为**public** 的子文件夹 ，这个文件夹中的文件如下图，你的 **.md 文件就被转换成了静态的网页文件，就是 index.html 还有一些css格式 文件之类的。对于每一次你写一篇文章 都要使用 `hexo g` 命令生成这个文件。夹。

  ![1508658592308](../pictures/images-hexo_blog/1508658592308.png)

- `hexo d` 命令是将 **public** 文件夹中的内容推送到远程仓库，就是你的这个项目 **your_username.github.io** 。 这个命令相当于 `git push` 命令。





`hexo clean` 命令清除也是清除的 **public** 文件夹。因为每次你发布文章，使用 `hexo g` 生成的  **public** 文件夹里的内容是一样的，等里面内容通过 `hexo d` 命令推送到 github 仓库后就没用了。



推送完之后刷新一下你的仓库也就是这个 **your_username.github.io** ，可以看到一篇文章已经推送到github的仓库里了，这时访问 **your_username.github.io** 这个域名就可以打开你的博客了。 下图是 hexo 里面自带的一个 helloworld 的文章主页，主要用于演示原理。

![1508659187787](../pictures/images-hexo_blog/1508659187787.png)

![1508659835559](../pictures/images-hexo_blog/1508659835559.png)



到此，hexo +github 的博客框架已经搭建好了，但是比较丑，需要进一步美化，忽然发现网上已经很多人写了，这里也不再写了。推荐两篇：

[基于Hexo搭建博客并部署到Github Pages](http://www.cnblogs.com/sun-haiyu/p/7027093.html)

[Hexo搭建博客教程](https://thief.one/2017/03/03/Hexo%E6%90%AD%E5%BB%BA%E5%8D%9A%E5%AE%A2%E6%95%99%E7%A8%8B/)

https://github.com/limedroid/HexoLearning

[Hexo 3.1.1 静态博客搭建指南](http://lovenight.github.io/2015/11/10/Hexo-3-1-1-%E9%9D%99%E6%80%81%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/?)

基本上按照上述博客，做完就差不多了。接下来稍微说下我在搭建的过程中碰到的问题。



# 4 基本操作

## 4.2 图标设置

以菜单栏为例：

```yaml
menu:
  home: /home/ || home
  about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  machine_Learning: /machine_Learning || plane
  deep_Learning: /deep_Learning || rocket
  python: /python || bug
  archives: /archives/ || archive
```

在  [Font Awesome](http://fontawesome.io/icons/)  上选取感兴趣的图标，直接写在 `||` 之后，比如我写的 plane，rocket。

> Value before `||` delimeter is the target link.
>
> Value after `||` delimeter is the name of FontAwesome icon. If icon (with or without delimeter) is not specified, question icon will be loaded.



## 4.3 文章分类

**新增导航**

引用：http://www.cnblogs.com/sun-haiyu/p/7027093.html

> 默认导航栏只有首页、归档、标签、分类四项。如果想增加其他如C++、随笔等。需要打开`\themes\next\_config.yml`找到如下
>
> ```
> # When running the site in a subdirectory (e.g. domain.tld/blog), remove the leading slash (/archives -> archives)
> menu:
>   home: /
>   categories: /categories
>   tags: /tags
>   archives: /archives
>   # 这里是新增的，程序猿是一级目录，C是二级目录，同理随笔是一级目录
>   c++: /categories/程序猿/C/
>   python: /categories/程序猿/Python/
>   essay: /categories/随笔/
>   # 注意这里没有/categories
>   about: /about
> ```
>
> 假如我想新建C++、Python、随笔三个导航按钮，并且打开他们的效果如下图。
>
> ![img](http://obvjfxxhr.bkt.clouddn.com/hexo%E9%83%A8%E7%BD%B2%E5%88%B0github_360%E5%8F%8D%E9%A6%88%E6%84%8F%E8%A7%81%E6%88%AA%E5%9B%BE161908239992124.png)
>
> 需要注意的是前面要加上`/categories`，格式是这样`python: /categories/这里是文章的一级目录/这里是文章的二级目录/`。结尾要加上`/`分隔符。**这几个页面是不需要通过hexo new page来生成的。**关于导航栏及侧栏所用的图标来自[fontawesome](http://fontawesome.io/)。在`\themes\next\_config.yml`里配置。

![1509285887284](../pictures/images-hexo_blog/1509285887284.png)

也就是说上图中箭头两端是指向一个链接的。所以左侧边栏是不需要通过 hexo new page来再生成 一个 机器学习的page。



但是这里注意一点，假如说你在 `/categories` 目录下有一个 机器学习的分类，这时有篇文章是属于机器学习的，那么在markdown文件中，程序员一般会这么写categories: machine_learning，如下所示。

```yaml
title: 支持向量机
categories: machine_learning
```

![1509285643194](../pictures/images-hexo_blog/1509285643194.png)

那在 图上也是正常的，但是如果你点进链接：

![1509285714082](../pictures/images-hexo_blog/1509285714082.png)

有没有发现什么，下划线变成了中划线，按道理来说应该也是下划线才对。这时如果你在主题配置文件中写的是这样的：machine_learning: /categories/machine_learning || plane，就是下面这样：

```yaml
menu:
  home: /home/ || home
  about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  paper-note: /categories/paper-note || book
  machine_learning: /categories/machine_learning || plane
  deep_learning: /categories/deep_learning || rocket
  python: /categories/python || bug
  archives: /archives/ || archive
```

![1509286023498](../pictures/images-hexo_blog/1509286023498.png)



也就是说，如果你从上图圈起来的地方点进去，是会找不到页面的，如下图：

![1509286082710](../pictures/images-hexo_blog/1509286082710.png)

这个很坑。我在搞的时候就遇上了，搞半天找不到问题，后面才发现是域名转换的时候，它把文章中的下划线转成了中划线。所以在文章中添加分类时，尽量不要包含下划线，像machine_learning 这样，直接写成中划线machine-learning，或者直接写空格也可以machine learning，他会自动转成中划线。

那在主题配置文件中，就要这样写：

```yaml
menu:
  home: /home/ || home
  about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  paper-note: /categories/paper-note || book
  machine-learning: /categories/machine-learning || plane
  deep-learning: /categories/deep-learning || rocket
  python: /categories/python || bug
  archives: /archives/ || archive
```

这样就不会出问题了。



# 5 公式问题

hexo对markdown公式支持很不好啊，光解决公式显示的问题就搞了好久。

刚部署好的hexo博客，喜出望外，终于可以写博客了，但是发现公式不显示，以下开始解决这个问题的征程。

## 5.1 方案一：使用插件

应用自:http://catx.me/2014/03/09/hexo-mathjax-plugin/

基本步骤如下：

> 在blog文件夹安装hexo-math
>
> ```shell
> $ npm install hexo-math --save
> ```
>
> 在blog文件夹中执行：
>
> ```shell
> $ hexo math install
> ```
>
> 在`_config.yml`文件中添加：
>
> ```yaml
> plugins:
> - hexo-math
> ```



好按照步骤做完之后，发现不起作用，又搜了搜，发现需要在 markdown 文件中加入下面这样的代码：

```html
<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
```

这一段代码最好放在 用于存储标题，标签，分类等YAML格式的信息后面，独立成一段。要是放在标题等YAML格式的信息之前可能会导致标题不显示的结果。

加入上述代码之后，有一点效果，但是效果差强人意，如下图，部分显示了，还有部分没显示。



![1509263293453](../pictures/images-hexo_blog/1509263293453.png)

**正常应该这样：**

![1509265208220](../pictures/images-hexo_blog/1509265208220.png)

网上查了一下发现根本原因是：

引用自：http://masikkk.com/article/hexo-13-MathJax/

> Hexo默认先使用hexo-renderer-marked引擎渲染MarkDown，然后再交给MathJax渲染。hexo-renderer-marked会把一些特殊的markdown符号转换为相应的html标签，比如在markdown语法中，下划线`_`代表斜体，会被转化为`<em>`标签，`\\`也会被转义成一个`\`。而类Latex格式书写的数学公式下划线`_`表示角标，`\\`表示公式换行，有特殊的含义，所以MathJax引擎在渲染数学公式的时候就会出错。



这个方法还有一个缺点，就是如果在`_config.yml`文件中添加了如下代码

```yaml
plugins:
- hexo-math
```

会导致`hexo server` 或者 `hexo s` 命令不起作用，然后把这个代码删去，又恢复正常了。出现这个问题时，让我迷惑了半天，还以为是 hexo 版本的问题。

因为[官网](https://hexo.io/zh-cn/docs/server.html)有这样一句话，如下

> Hexo 3.0 把服务器独立成了个别模块，您必须先安装 [hexo-server](https://github.com/hexojs/hexo-server) 才能使用。
>
> ```
> $ npm install hexo-server --save
> ```

然后我看了看我的版本就是 3.3.9，而且在 node_modules 文件夹里也存在 hexo-server 文件夹。 那看来应该不是这个问题，直到我看到了这个帖子，https://github.com/gyk001/hexo-qiniu-sync/issues/41，才发现是 上面那个plugins:代码的问题。把他删掉就好了。但是我的公式还是不能完全显示。



## 5.2 方案二：还是使用插件，稍有不同

应用自：https://monkey0105.github.io/2017/09/21/hexo-mathjax/

方法大同小异，基本步骤如下：

> **安装**
>
> ```shell
> npm install hexo-math --save
> ```
>
> **在站点配置文件中加入以下内容：**
>
> ```yaml
> math:
>   engine: 'mathjax' # or 'katex'
>   mathjax:
>     src: custom_mathjax_source
>     config:
>       # MathJax config
>   katex:
>     css: custom_css_source
>     js: custom_js_source # not used
>     config:
>       # KaTeX config
> ```
>
> **主题配置文件**
>
> 编辑Next的主题配置文件， 将`mathjax`下的`enable`设定为`true`即可。`cdn`用于指定 MathJax的脚本地址，默认是MathJax官方提供的CDN地址。
>
> ```yaml
> mathjax:
>   enable: true
>   cdn: //cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML
> ```



只不过最后的效果依然是：

![1509263293453](../pictures/images-hexo_blog/1509263293453.png)



还是 MarkDown和MathJax的语法冲突问题。

但我不知道的是为什么上面的就没问题，下面那个式子就不能正常显示，好奇怪，后来观察了一下不能正常显示的都是那些多行的公式，单行公式和行间公式是正常的。因为这个里面下面那个公式其实是一个公式，只是分成三行来写了。

![1509265208220](../pictures/images-hexo_blog/1509265208220.png)

## 5.3 方案三：将Hexo的hexo-renderer-marked渲染引擎换成 hexo-renderer-markdown-it-plus 

引用自：http://blog.cofess.com/2017/09/06/how-to-use-mathjax-to-render-latex-mathematical-formulas-in-hexo.html

基本用法：

> **安装** 
>
> 先卸载hexo-renderer-marked，再安装hexo-renderer-markdown-it-plus插件
>
> ```shell
> npm uninstall hexo-renderer-marked --save
> npm install hexo-renderer-markdown-it-plus --save
> ```
>
>  **配置**
>
> 安装插件后，如果未正常渲染LaTeX数学公式，在博客配置文件`_config.yml`中添加
>
> ```yaml
> markdown_it_plus:
>   highlight: true
>   html: true
>   xhtmlOut: true
>   breaks: true
>   langPrefix:
>   linkify: true
>   typographer:
>   quotes: “”‘’
>   plugins:
>     - plugin:
>         name: markdown-it-katex
>         enable: true
>     - plugin:
>         name: markdown-it-mark
>         enable: false
> ```
>
>  **文章启用mathjax**
>
> ```yaml
> title: Hello World
> mathjax: true
> ```

效果更差了：如下图还是那两个公式，那个多行公式还是没有正常显示，其他单行公式显示也不正常了，每个公式都被重复了一遍。

![1509265933076](../pictures/images-hexo_blog/1509265933076.png)

然后还出现了像下图这样的不知名的东东。。。

![1509265944549](../pictures/images-hexo_blog/1509265944549.png)



## 5.4 方案四：将Hexo的hexo-renderer-marked渲染引擎换成  hexo-renderer-pandoc

引用自：https://cethik.vip/2016/09/20/mathjaxSolve/

基本步骤：

> **首先**
>
> 你得在电脑中装好pandoc ,具体安装包在[这里](http://johnmacfarlane.net/pandoc/installing.html),下载完成后按照，一路next，安装完毕后打开电脑终端（windows命令提示符），保证`pandoc --help`这条命令可以运行才可以继续下一步
>
> **安装pandoc**
>
> 安装pandoc到你的hexo，首先命令行切换到你的hexo博客的根目录下，然后运行`npm install hexo-renderer-pandoc --save` , 当然如果你安装过程出了问题可以直接卸载掉`sudo npm uninstall hexo-renderer-pandoc`，OK，到这里基本上就OK了，然后运行
>
> ```
> hexo clean
> hexo g
> ```



然后我又看到了这个博客：http://yanghan.life/2017/07/01/hexo%E5%85%AC%E5%BC%8F%E6%98%BE%E7%A4%BA/，文中这样说：

> 公式显示我尝试换了`pandoc`的渲染，装了`pandoc`和`hexo-renderer-pandoc`，卸载了原装的`hexo-renderer-marked`，但是本地`hexo s`虽然显示正常，但是`deploy`过后网站上的就只有将`$$`转义成`\[`和`\]`的东西

所以我不打算尝试这种方法了。



## 5.5 方案五：直接修改hexo的 Markdown 渲染引擎

折腾了好一会儿，我还是直接打算使用最根本的解决方案。

首先按照方案二，做以下修改：

引用自：https://monkey0105.github.io/2017/09/21/hexo-mathjax/

> **安装**
>
> ```shell
> npm install hexo-math --save
> ```
>
> **在站点配置文件中加入以下内容：**
>
> ```yaml
> math:
>   engine: 'mathjax' # or 'katex'
>   mathjax:
>     src: custom_mathjax_source
>     config:
>       # MathJax config
>   katex:
>     css: custom_css_source
>     js: custom_js_source # not used
>     config:
>       # KaTeX config
> ```
>
> **主题配置文件**
>
> 编辑Next的主题配置文件， 将`mathjax`下的`enable`设定为`true`即可。`cdn`用于指定 MathJax的脚本地址，默认是MathJax官方提供的CDN地址。
>
> ```yaml
> mathjax:
>   enable: true
>   cdn: //cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML
> ```





然后做以下修改：

引用自：http://blog.csdn.net/emptyset110/article/details/50123231

> **使Marked.js与MathJax共存**
>
> 因此我提供一个修改marked.js源码的方式来避开这些问题 
> \- 针对下划线的问题，我决定取消`_`作为斜体转义，因为marked.js中`*`也是斜体的意思，所以取消掉`_`的转义并不影响我们使用markdown，只要我们习惯用`*`作为斜体字标记就行了。 
> \- 针对marked.js与Mathjax对于个别字符二次转义的问题，我们只要不让marked.js去转义`\\`,`\{`,`\}`在MathJax中有特殊用途的字符就行了。 
> 具体修改方式，用编辑器打开marked.js（在`./node_modules/marked/lib/`中）
>
> **Step 1:**
>
> ```
>   escape: /^\\([\\`*{}\[\]()# +\-.!_>])/,1
> ```
>
> 替换成
>
> ```
>   escape: /^\\([`*\[\]()# +\-.!_>])/,1
> ```
>
> **这一步是在原基础上取消了对`\\`,`\{`,`\}`的转义(escape)**
>
> **Step 2:**
>
> ```
>   em: /^\b_((?:[^_]|__)+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,1
> ```
>
> 替换成
>
> ```
>   em:/^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,1
> ```
>
> 这样一来MathJax就能与marked.js共存了。重启一下hexo看一下吧



重启之后，果然有惊喜，看下图，不管行内的，行间的，单行的，多行的，全都显示出来了，至此问题解决。

![1509267319356](../pictures/images-hexo_blog/1509267319356.png)





感谢各位大佬提供的解决方案，让我深受启发，想从源码层面了解的还可以看[这个](http://masikkk.com/article/hexo-13-MathJax/) 。这里就不解释了。

# 6 永久链接

参考这两个：

https://post.zz173.com/detail/hexo-abbrlink.html

https://www.npmjs.com/package/hexo-abbrlink



> ## How to install
>
> Add plugin to Hexo:
>
> ```bash
> npm install hexo-abbrlink --save
> ```
>
> Modify permalink in config.yml file:
>
> ```yaml
> permalink: posts/:abbrlink/
> ```
>
> There are two settings:
>
> ```yaml
> alg -- Algorithm (currently support crc16 and crc32, which crc16 is default)
> rep -- Represent (the generated link could be presented in hex or dec value)
> # abbrlink config
> abbrlink:
>   alg: crc32  #support crc16(default) and crc32
>   rep: hex    #support dec(default) and hex
> ```

安装的时候 出错：

```shell
$ npm install hexo-abbrlink --save
hexo-site@0.0.0 D:\myBlog
`-- hexo-abbrlink@2.0.5

npm WARN optional SKIPPING OPTIONAL DEPENDENCY: fsevents@^1.0.0 (node_modules\chokidar\node_modules\fsevents):
npm WARN notsup SKIPPING OPTIONAL DEPENDENCY: Unsupported platform for fsevents@1.1.3: wanted {"os":"darwin","arch":"any"} (current: {"os":"win32","arch":"x64"})
```

维基百科：

> **Darwin**是由[苹果公司](https://zh.wikipedia.org/wiki/%E8%98%8B%E6%9E%9C%E5%85%AC%E5%8F%B8)于2000年所发布的一个[开放源代码](https://zh.wikipedia.org/wiki/%E9%96%8B%E6%94%BE%E5%8E%9F%E5%A7%8B%E7%A2%BC)[操作系统](https://zh.wikipedia.org/wiki/%E4%BD%9C%E6%A5%AD%E7%B3%BB%E7%B5%B1)。Darwin是[Mac OS X](https://zh.wikipedia.org/wiki/Mac_OS_X)和[iOS](https://zh.wikipedia.org/wiki/IOS)操作环境的操作系统部分。苹果公司于2000年把Darwin发布给开放源代码社区。

fsevents是mac下用的，windows不支持，反正是一个依赖项，而且也只是一个警告，不用管它。

想解决的话，参考这里： https://github.com/angular/angular/issues/13935 

使用下面命令安装即可：

```shell
npm install --no-optional hexo-abbrlink --save
```

安装好之后，hexo clean 一下，再重新生成。