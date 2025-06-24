# Pref-LSTM

Memory storage for Large Language models (LLMs) is
becoming an increasingly active area of research, partic-
ularly for enabling personalization across long conversa-
tions. We propose Pref-LSTM, a dynamic and lightweight
framework that combines a BERT-based classifier with
a LSTM memory module that generates memory embed-
ding which then is soft-prompt injected into a frozen LLM.
We synthetically curate a dataset of preference and non-
preference conversation turns to train our BERT-based
classifier. Although our LSTM-based memory encoder did
not yield strong results, we find that the BERT-based clas-
sifier performs reliably in identifying explicit and implicit
user preferences. Our research demonstrates the viability
of using preference filtering with LSTM gating principals
as an efficient path towards scalable user preference mod-
eling, without extensive overhead and fine-tuning.
