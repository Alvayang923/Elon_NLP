An Exploratory Analysis of Elon Musk’s Communication on Twitter
=============
NLP project analyzing Elon Musk Twitter activity.

Contributors
------------
- Ludmila Filipova(https://github.com/filipol2)
- Renfang Yang(https://github.com/Alvayang923)
- Pengyun Li(https://github.com/whale9707)
- Alana Jean Reinert(https://github.com/AJR323)
- Jan Batzner

简介
=============
据美国财富杂志《福布斯》报道，特斯拉首席执行官埃隆·马斯克(Elon Musk)以2190亿美元净资产成为全球首富。他是电子支付公司PayPal的联合创始人，是运载火箭和航天器制造商 SpaceX创始人，同时，他又是推特上粉丝最多的明星企业家之一，目前粉丝数已超过一亿。人们紧盯他的账号，因为马斯克的一条推特可能带来各种“蝴蝶效应“，影响着股价、加密货币价格甚至其他社会经济现象。

因此，我们的项目将基于埃隆·马斯克在2010-2022年的Twitter推文进行探索性分析。我们首先使用自然语言处理技术进行文本数据的清理，并基于token进行探索性数据分析。根据探索性数据分析结果，我们利用一致性分数(Coherence Score)和LDA(Latent Dirichlet allocation)主题模型进行主题建模分析。此外，我们还使用VADER对推文进行情感分析。

通过我们的研究，希望能够从埃隆·马斯克的推文中揭示更多有趣的趋势，帮助我们更好地理解科技领域变化趋势，并为进一步研究提供有价值的参考。


涉及方法
=============

#### 1. 数据处理:

- Emojis to words -> pre-processing -> stemming -> stop words

#### 2. 数据分析:

- 探索性分析 (词频，高频话题标签，发文量，发文时间等)
- LDA主题模型
- 情感分析

数据源
=============
Dalila, A. Kaggle数据集 [Elon Musk Tweets (2010 - 2022)](https://www.kaggle.com/datasets/ayhmrba/elon-musk-tweets-2010-2021) 


Introduction
=============
Elon Musk, who is the current world's richest man, according to Forbes, was the co-founder of  the electronic-payment company PayPal and the founder of  SpaceX, which is the maker of launch vehicles and spacecraft. He was also one of the first significant investors in, as well as chief executive officer of, the electric car manufacturer Tesla. In recent years, Elon has founded Neurolink and OpenAI to dive deeper into the areas of artificial intelligence. Being such a prominent  influencer, the words of Elon can be significant signals to help us know better about what is going on in the world. 

Our team therefore decided to conduct an exploratory analysis of Elon Musk’s communication on Twitter from 2010 to 2022. We started with processing the Tweets with natural language processing techniques followed by a thorough exploratory data analysis (EDA) on the tokens. Based on findings from our EDA, we conducted an in-depth topic modeling analysis using the coherence score and the latent dirichlet allocation (LDA) for interpretations. Besides, the team also dives into the sentiment analysis of Elon’s Tweets with VADER (Valence Aware Dictionary and sEntiment Reasoner) to find any insightful pattern. 

We wish that our work can provide valuable references and analyses for any potential further research aiming for revealing more interesting patterns. 


Techniques
=============

#### 1. Data Processing:

- Emojis to words -> pre-processing -> stemming -> stop words

#### 2. Data Analysis:

- Exploratory analysis (Top hashtags & mentions, word frequency, monthly number of posts, timezone/favorite posting times etc.)
- LDA Topic Modeling
- Sentiment Analysis

Data Source
=============
The Kaggle dataset [Elon Musk Tweets (2010 - 2022)](https://www.kaggle.com/datasets/ayhmrba/elon-musk-tweets-2010-2021) collected and regularly updated by Ayhm Dalila


