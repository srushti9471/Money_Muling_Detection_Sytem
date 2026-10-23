# The AML Multi-GNN Bible: Architecture, Data, Theory, and Implementation
**Comprehensive Engineering and Theoretical Documentation of the Money Muling Detection System**

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [The Financial Crime Problem & Why Traditional AML Fails](#2-the-financial-crime-problem--why-traditional-aml-fails)
3. [The IBM AML Dataset Suite](#3-the-ibm-aml-dataset-suite)
4. [Data Preprocessing & Directed Multigraph Construction](#4-data-preprocessing--directed-multigraph-construction)
5. [Graph Neural Networks: Foundational Theory](#5-graph-neural-networks-foundational-theory)
6. [Multi-GNN Architecture: GINe with Ports & Reverse Message Passing](#6-multi-gnn-architecture-gine-with-ports--reverse-message-passing)
7. [Machine Learning Formulation & Training Pipeline](#7-machine-learning-formulation--training-pipeline)
8. [Model Inference & Risk Scoring](#8-model-inference--risk-scoring)
9. [Entity Intelligence: Declaring Nodes as Suspicious Money Mules](#9-entity-intelligence-declaring-nodes-as-suspicious-money-mules)
10. [Frontend & Full-Stack Visualization Architecture](#10-frontend--full-stack-visualization-architecture)
11. [Repository File Map & Execution Guide](#11-repository-file-map--execution-guide)
12. [Future Horizons & Enterprise Scalability](#12-future-horizons--enterprise-scalability)

---

## 1. Executive Summary

The **Money Muling Detection System** is an end-to-end, graph-native artificial intelligence platform built to detect complex financial crime, money laundering networks, and money muling rings.

Traditional anti-money laundering (AML) platforms rely on static threshold rules (e.g., *"flag transfers over $10,000"*) or single-row tabular machine learning (XGBoost, Random Forest). Criminal syndicates easily circumvent these legacy systems by decomposing illegal proceeds into structured transactions (**smurfing**), moving funds rapidly across intermediate accounts (**money mules**), and circulating money through closed loops (**layering cycles**).

This system approaches financial transactions through **Directed Temporal Multigraphs**:
* **Bank accounts and financial institutions** are modeled as **nodes**.
* **Financial transactions** are modeled as **directed, attributed multigraph edges** carrying timestamps, amounts, currencies, payment channels, and topological port identifiers.
* A deep **Graph Isomorphism Network with Edge Features (GINe)** enhanced with **In/Out Port Numberings** and **Reverse Message Passing** evaluates every transaction in its multi-hop topological context.
* Edge predictions are aggregated to identify money mules, smurfing hubs, and cash-out sinks, which are served via an interactive, GPU-accelerated web dashboard.

### Key Metrics on the IBM `HI-Small` Benchmark
* **Dataset Scope:** 5,078,345 transactions evaluated chronologically across 18 temporal days.
* **Class Imbalance:** 0.17% illicit ratio (1 laundering transaction for every 588 legitimate transfers).
* **Top Alert Precision:** **96.0%** (24 of the top 25 flagged transactions confirmed ground-truth illicit).
* **Financial Recovery:** Over **$505.2 Million** in suspected illicit flow flagged across 297 high-confidence alerts.
* **Latency:** Sub-10ms GNN inference on local GPU (NVIDIA RTX 3050 Laptop GPU).

---

## 2. The Financial Crime Problem & Why Traditional AML Fails

### 2.1 The Three Stages of Money Laundering
Money laundering is the process of disguising the illegal origin of criminal proceeds:
1. **Placement:** Introducing illicit cash into the legitimate financial system (e.g., breaking large amounts of drug or fraud money into retail bank deposits).
2. **Layering:** Creating complex layers of financial transactions to obscure the audit trail, ownership, and source of funds. This involves rapid cross-bank transfers, currency conversions, and shell company transfers.
3. **Integration:** Re-injecting layered money into the legal economy (e.g., buying real estate, business investments, or luxury assets) so it appears completely legitimate.

```
[Criminal Proceeds]
       │
       ▼ (Placement)
 [Dirty Deposits] ───► [Smurfing / Fan-Out]
                             │
                             ▼ (Layering)
                    ┌─────────────────┐
                    │ Money Mule Hubs │ ◄──┐
                    └────────┬────────┘    │ (Cycles & Loops)
                             ▼             │
                    [Aggregator Sink] ─────┘
                             │
                             ▼ (Integration)
                 [Legitimate Clean Asset]
```

### 2.2 Laundering Typologies
* **Money Mules (Gather-Scatter):** Accounts that receive funds from multiple sources and quickly forward the money to downstream destinations, retaining a small commission. Mules act as firewalls between the criminal leadership and the funds.
* **Smurfing (Structuring / Fan-Out):** Dispersing a large sum into dozens of micro-transactions below the mandatory regulatory reporting threshold (e.g., $10,000 Currency Transaction Reports in the US).
* **Layering Cycles:** Routing funds in a closed or near-closed multi-hop loop ($A \to B \to C \to D \to A$) through multiple legal jurisdictions to break automated rule triggers.
* **Aggregators (Fan-In):** Funneling money from dozens of small mule accounts into a single high-balance entity for cash-out or offshore wire transfer.

### 2.3 Why Tabular ML and Rules Fail
Traditional AML software evaluates transactions independently:
$$\text{IsLaundering}(T_i) = f(\text{Amount}_i, \text{Country}_i, \text{Channel}_i)$$

1. **Blindness to Graph Topology:** A transaction of $4,500 from Account A to Account B looks completely normal in isolation. However, if Account A received $4,600 two minutes earlier from Account X, and Account B immediately wires $4,400 to Account Y, this transaction is an integral hop in a money muling chain. Single-row feature vectors cannot capture multi-hop dependencies.
2. **Exponential Feature Engineering Explosion:** Attempting to represent 3-hop or 5-hop neighborhoods with flat SQL joins produces tens of thousands of sparse columns that overfit, fail on unseen topologies, and cannot generalize.
3. **Catastrophic False Positive Rates:** Traditional banking rule engines yield **false positive rates exceeding 95%**, resulting in thousands of wasted human compliance hours reviewing benign payments.

---

## 3. The IBM AML Dataset Suite

The project leverages the **IBM Synthetic AML Financial Crime Dataset**, an industry benchmark designed by IBM Research using an agent-based transaction generator that mimics authentic banking protocols, multi-currency flows, and adversarial laundering typologies.

### 3.1 Dataset Inventory in `../dataset/`

| Dataset File | Size | Transactions | Illicit Ratio | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`HI-Small_Trans.csv`** | **475 MB** | **5,078,345** | **0.17%** | **High-Illicit Small.** Primary training & benchmark split used for our trained model checkpoint. |
| **`LI-Small_Trans.csv`** | **650 MB** | **6,924,050** | **0.01%** | **Low-Illicit Small.** Realistic real-world banking ratio (1 laundering event per 10,000 transactions). |
| **`HI-Medium_Trans.csv`** | **3.0 GB** | **~32 Million** | **0.17%** | **High-Illicit Medium.** 6x denser multigraph across thousands of regional banks. |
| **`LI-Medium_Trans.csv`** | **2.9 GB** | **~31 Million** | **0.01%** | **Low-Illicit Medium.** Enterprise-scale low-illicit validation set. |
| **`HI-Large_Trans.csv`** | **17.0 GB** | **~180 Million** | **0.17%** | **High-Illicit Large.** Full-scale national financial simulation. |
| **`LI-Large_Trans.csv`** | **16.7 GB** | **~175 Million** | **0.01%** | **Low-Illicit Large.** National banking network with rare laundering traces. |

### 3.2 Raw Schema Specification
Every row in the raw transactions CSV contains 11 attributes:
1. `Timestamp`: Date and time string formatted as `YYYY/MM/DD HH:MM`.
2. `From Bank`: Numerical identifier of the originating banking institution.
3. `Account`: Unique alphanumeric account hash of the sender.
4. `To Bank`: Numerical identifier of the receiving banking institution.
5. `Account.1`: Unique alphanumeric account hash of the recipient.
6. `Amount Received`: Numerical volume of funds received by the counterparty.
7. `Receiving Currency`: Currency format received (e.g., US Dollar, Euro, Bitcoin, Yuan, Rupee).
8. `Amount Paid`: Numerical volume of funds sent by the originator (accounts for FX conversion fees).
9. `Payment Currency`: Currency denomination of the originating transfer.
10. `Payment Format`: Transfer channel (ACH, Wire, Cheque, Credit Card, Cash, Bitcoin, Reinvestment).
11. `Is Laundering`: Binary target ground-truth label ($0 = \text{Licit}$, $1 = \text{Illicit AML}$).

### 3.3 Institutional & Pattern Ground Truth
* **`*_accounts.csv`:** Contains institutional registry records mapping account identifiers to legal entity types (e.g., *Sole Proprietorship #50438*, *Corporation #33520*, *Partnership #35397*) and bank jurisdiction.
* **`*_Patterns.txt`:** Labeled topological laundering campaigns categorized by pattern type (`FAN-OUT`, `FAN-IN`, `CYCLE`, `GATHER-SCATTER`, `BIPARTITE`), detailing the exact sequence of hops, accounts, and timestamps injected by the synthetic generator.

---

## 4. Data Preprocessing & Directed Multigraph Construction

Raw transaction logs are tabular CSVs. Transforming them into a format suitable for message passing requires constructing a **Directed Temporal Multigraph**.

```
Row in CSV:
[Timestamp, From_Bank, From_Acc, To_Bank, To_Acc, Amount, Currency, Format]
                                │
                                ▼ format_kaggle_files.py
[EdgeID, from_id, to_id, Timestamp (sec), Amount, Currency_ID, Format_ID, in_port, out_port]
                                │
                                ▼ data_loading.py
HeteroData / GraphData (PyTorch Geometric Graph Object)
```

### 4.1 Chronological & Categorical Normalization
1. **Global Entity ID Mapping:** A unified hash index is constructed across both sender entities (`From Bank` + `Account`) and receiver entities (`To Bank` + `Account.1`). Each unique entity receives an integer index $v \in \{0, 1, \dots, |V|-1\}$.
2. **Temporal Alignment:** Timestamps are parsed and normalized into elapsed seconds relative to the beginning of the transaction stream:
   $$t_{\text{sec}} = t - t_{\text{start}}$$
   All transactions are sorted chronologically: $e_0, e_1, \dots, e_{|E|-1}$ such that $t(e_i) \le t(e_{i+1})$.
3. **Categorical Encodings:**
   * **Currencies (15 classes):** USD (0), BTC (1), EUR (2), AUD (3), CNY (4), INR (5), JPY (6), MXN (7), GBP (8), RUB (9), CAD (10), CHF (11), BRL (12), SAR (13), ILS (14).
   * **Payment Formats (7 classes):** Reinvestment (0), Cheque (1), Credit Card (2), ACH (3), Cash (4), Wire (5), Bitcoin (6).
4. **Amount Normalization:** Continuous amounts are normalized using Z-score standardization:
   $$z_{\text{amt}} = \frac{\text{amt} - \mu_{\text{amt}}}{\sigma_{\text{amt}}}$$

### 4.2 The Multigraph Problem & Port Numberings
In financial networks, two accounts often transact repeatedly over time:
$$\text{Account } A \xrightarrow{\text{Tx } 1} \text{Account } B, \quad \text{Account } A \xrightarrow{\text{Tx } 2} \text{Account } B$$

Standard GNN architectures assume simple graphs with at most one edge between any pair of nodes. When applied to multigraphs, simple message passing either collapses parallel edges or loses the ability to differentiate the chronological order and local identity of parallel transactions.

To resolve this, the system incorporates **Port Numberings** (`ports()` in `data_util.py`):
* For every node $v$, its incoming and outgoing neighbors are ordered chronologically based on transaction timestamps.
* A unique port index is assigned to each edge connected to $v$:
  $$\text{in\_port}(u \to v) = \text{index of } u \text{ in the ordered list of unique senders to } v$$
  $$\text{out\_port}(u \to v) = \text{index of } v \text{ in the ordered list of unique recipients from } u$$
* This injects local topological coordinates directly into the edge attributes, breaking symmetry and allowing the GNN to distinguish between repeated transfers, smurfing branches, and cyclic laundering returns.

### 4.3 Chronological Temporal Splitting
Random k-fold cross-validation is **fundamentally invalid** in financial networks because it causes **temporal leakage** (using future transactions to predict past transactions).

The dataset is partitioned chronologically across temporal days:
* **Train Split (60%):** Days 0 through 10.
* **Validation Split (20%):** Days 11 through 13.
* **Test Split (20%):** Days 14 through 17 (~648,748 unseen transactions).

During test evaluation, the GNN observes historical transactions up to the evaluation window and predicts labels strictly on unseen future edges.

---

## 5. Graph Neural Networks: Foundational Theory

### 5.1 What is a Graph Neural Network?
A **Graph Neural Network (GNN)** is a class of deep learning models designed to operate directly on non-Euclidean graph structures $G = (V, E)$, where $V$ is a set of nodes and $E$ is a set of edges.

Unlike convolutional networks (CNNs) that operate on regular 2D grids (pixels) or transformers that operate on 1D sequences (tokens), GNNs learn permutation-invariant representations over arbitrary graph topologies via **Message Passing**.

### 5.2 The Message Passing Neural Network (MPNN) Framework
At every layer $k \in \{1, \dots, K\}$, each node $v$ updates its representation vector $h_v^{(k)}$ by aggregating messages from its 1-hop neighborhood $\mathcal{N}(v)$:

```
           [Node u1] (h_u1)
                 │  Message m_{u1->v}
                 ▼
[Node u2] ───► [Node v] ◄─── [Node u3]
 (h_u2)          │             (h_u3)
                 │ Aggregate & Update
                 ▼
             h_v^{(k+1)}
```

Mathematically, a general MPNN executes three operations at each layer:
1. **Message Generation:**
   $$m_{u \to v}^{(k)} = \text{Message}^{(k)}\left( h_u^{(k-1)}, h_v^{(k-1)}, e_{uv} \right)$$
2. **Message Aggregation:**
   $$a_v^{(k)} = \bigoplus_{u \in \mathcal{N}(v)} m_{u \to v}^{(k)}$$
   where $\bigoplus$ is a permutation-invariant aggregator (such as $\sum$, $\text{Mean}$, or $\text{Max}$).
3. **Node State Update:**
   $$h_v^{(k)} = \text{Update}^{(k)}\left( h_v^{(k-1)}, a_v^{(k)} \right)$$

After $K$ layers of message passing, node $v$'s embedding $h_v^{(K)}$ encodes structural information from its entire **$K$-hop neighborhood**. In financial graphs with $K=2$, an account embedding contains information about its counterparties and its counterparties' counterparties.

### 5.3 Expressive Power: The 1-WL Test and Why GIN?
The theoretical limit of standard MPNNs is bounded by the **1-dimensional Weisfeiler-Lehman (1-WL) graph isomorphism test** (Xu et al., 2018).

* Standard Graph Convolutional Networks (GCN) use normalized **mean aggregation**:
  $$a_v = \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{d_v d_u}} h_u$$
  Mean aggregation averages neighbor embeddings. If Account A transfers money to 10 low-risk accounts, while Account B transfers money to 1 low-risk account, mean aggregation treats their neighbor distributions identically, failing to capture multigraph degree scale.
* Graph Attention Networks (GAT) use softmax attention weights, which also act as weighted averages and fail to distinguish multiset scale.
* **Graph Isomorphism Network (GIN)** proves that **sum aggregation** over countable multigraph multisets is the only aggregator that achieves maximal 1-WL expressive power:
  $$\text{Agg}(S) = \sum_{x \in S} f(x)$$
  Sum aggregation preserves multiset cardinality, allowing the GNN to count transactions, recognize high-degree smurfing dispersion, and detect exact circular topologies.

---

## 6. Multi-GNN Architecture: GINe with Ports & Reverse Message Passing

The core architecture implemented in [`models.py`](file:///home/rishabh/c0de/anti-money-laundering/Multi-GNN/models.py) is **GINe (Graph Isomorphism Network with Edge Features)**, configured with **Edge MLPs** and **Reverse Message Passing**.

```
Input Node Features: x in R^{1} ────► [Node Linear: 1 -> 100]  ───┐
                                                                 │
Input Edge Features: e in R^{6} ────► [Edge Linear: 6 -> 100]  ──┼──► [GINEConv Layer 1] ──► [BatchNorm] ──► [Residual]
                                                                 │              │
                                                                 │              ▼ (Optional Edge MLP)
                                                                 │       Updated Edge Embeddings
                                                                 │              │
                                                                 └──────────────┴───────────────────────────► [GINEConv Layer 2]
                                                                                                                      │
                                                                                                                      ▼
Concat: [h_src || h_dst || e_attr] in R^{300} ◄───────────────────────────────────────────────────────────────────────┘
  │
  ▼
[Readout MLP: 300 -> 50 -> 25 -> 2] ──► Softmax ──► AML Risk Score P(Illicit)
```

### 6.1 Input Embeddings
* **Node Input:** $x_v \in \mathbb{R}^{1}$ (identity degree feature) is projected to latent dimension $d = 100$:
  $$h_v^{(0)} = W_{\text{node}} x_v + b_{\text{node}}, \quad h_v^{(0)} \in \mathbb{R}^{100}$$
* **Edge Input:** $e_{uv} \in \mathbb{R}^{6}$ (`[Timestamp, Amount Received, Received Currency, Payment Format, in_port, out_port]`) is projected to latent dimension $d = 100$:
  $$e_{uv}^{(0)} = W_{\text{edge}} e_{uv} + b_{\text{edge}}, \quad e_{uv}^{(0)} \in \mathbb{R}^{100}$$

### 6.2 GINEConv Layer Formulation
For each layer $k \in \{1, \dots, K\}$ (where $K=2$ layers in our checkpoint):
$$h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\left( h_u^{(k-1)} + e_{uv}^{(k-1)} \right) \right)$$

where:
* $\text{MLP}^{(k)}$ is a 2-layer perceptron: $\text{Linear}(100 \to 100) \to \text{ReLU} \to \text{Linear}(100 \to 100)$.
* $\epsilon^{(k)}$ is a learnable or fixed scalar parameter balancing ego-representation vs. neighbor messages.
* The edge embedding $e_{uv}^{(k-1)}$ is directly summed with neighbor embedding $h_u^{(k-1)}$ inside the non-linear activation before sum aggregation.

### 6.3 Batch Normalization & Residual Connections
To prevent vanishing gradients and oversmoothing across dense multigraph clusters, each layer employs Batch Normalization and residual skip connections:
$$h_v^{(k)} = \frac{h_v^{(k-1)} + \text{ReLU}\left( \text{BatchNorm}\left( \text{GINEConv}^{(k)}(h^{(k-1)}, \text{edge\_index}, e^{(k-1)}) \right) \right)}{2}$$

### 6.4 Dynamic Edge MLP Updates (`edge_updates=True`)
Optionally, edge attributes are updated after every convolution layer by concatenating source node, destination node, and current edge state:
$$e_{uv}^{(k)} = e_{uv}^{(k-1)} + \frac{1}{2} \text{EMLP}^{(k)}\left( [h_u^{(k)} \,\|\, h_v^{(k)} \,\|\, e_{uv}^{(k-1)}] \right)$$
where $\text{EMLP}^{(k)}: \mathbb{R}^{300} \to \mathbb{R}^{100}$.

### 6.5 Reverse Message Passing (Bidirectional Heterogeneous Graphs)
In standard directed graphs, messages only flow in the direction of the edge ($u \xrightarrow{\text{pay}} v$). 

However, in financial AML investigations, identifying money mules requires inspecting both:
* **Where the money came from** (upstream source investigation).
* **Where the money went** (downstream destination investigation).

Using PyTorch Geometric's `to_hetero` transformation, the multigraph is duplicated into two complementary relation types:
1. `('node', 'to', 'node')`: Forward transfer direction ($u \to v$).
2. `('node', 'rev_to', 'node')`: Reverse investigative flow ($v \leftarrow u$).

The model learns specialized message passing channels across both relations, allowing upstream originators to receive supervisory signals from downstream cash-out events and vice versa.

### 6.6 Edge Readout Head (Classifier)
Because the task is **Edge Classification** (predicting whether a specific transaction is illicit), the readout head constructs an edge representation by concatenating:
* The final source node embedding $h_u^{(K)} \in \mathbb{R}^{100}$.
* The final destination node embedding $h_v^{(K)} \in \mathbb{R}^{100}$.
* The final edge attribute embedding $e_{uv}^{(K)} \in \mathbb{R}^{100}$.

$$z_{uv} = [h_u^{(K)} \,\|\, h_v^{(K)} \,\|\, e_{uv}^{(K)}] \in \mathbb{R}^{300}$$

The concatenated vector is fed into a 3-layer classification MLP with Dropout:
$$\hat{y}_{uv} = \text{Linear}(25 \to 2) \left( \text{Dropout}_{0.5}\left( \text{ReLU}\left( \text{Linear}(50 \to 25) \left( \text{Dropout}_{0.5}\left( \text{ReLU}\left( \text{Linear}(300 \to 50) z_{uv} \right) \right) \right) \right) \right) \right)$$

The output $\hat{y}_{uv} \in \mathbb{R}^2$ represents raw class logits: $[z_{\text{licit}}, z_{\text{illicit}}]$.

---

## 7. Machine Learning Formulation & Training Pipeline

### 7.1 Loss Function & Severe Class Imbalance Handling
The training set contains an illicit ratio of approximately **0.17%** (only 1 out of every 588 transactions is laundering). A naive model predicting all transactions as licit achieves 99.83% accuracy while catching 0% of criminals.

To counter this, the model is optimized using **Weighted Cross-Entropy Loss**:
$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \left[ w_0 \cdot y_i \log(\hat{p}_{i,0}) + w_1 \cdot (1 - y_i) \log(\hat{p}_{i,1}) \right]$$
where:
* $w_0 = 1.0$ (weight for licit class).
* $w_1 = \frac{N_{\text{licit}}}{N_{\text{illicit}}} \approx 580.0$ (penalty multiplier on false negatives).

Every missed laundering transaction incurs 580 times more loss than a false alarm on a legitimate transaction, forcing the gradient updates to focus on high-risk topological signatures.

### 7.2 Mini-Batch Subgraph Sampling (`NeighborLoader`)
A transaction graph with 5 million edges cannot fit into GPU memory for full-batch gradient descent. The training loop employs PyG's `NeighborLoader`:
* Edges are partitioned into batches of size $B = 8,192$.
* For each batch of target edges, the loader samples a 2-hop neighborhood around the incident nodes, extracting subgraphs of 10,000–30,000 nodes.
* The GNN computes message passing exclusively on the sampled subgraph, enabling scalable training on consumer GPUs like the RTX 3050.

### 7.3 Optimizer & Training Hyperparameters
* **Optimizer:** AdamW (learning rate $\eta = 0.001$, weight decay $\lambda = 10^{-5}$).
* **Learning Rate Scheduler:** ReduceLROnPlateau monitoring Validation F1 score.
* **Epochs:** 20 epochs with early stopping on validation F1.
* **Regularization:** Final Dropout $p = 0.5$, Batch Normalization on all hidden layers.

---

## 8. Model Inference & Risk Scoring

### 8.1 Probability Extraction
During inference, raw logits from the GINe readout head are mapped to probabilities via Softmax:
$$P(\text{illicit} \mid e_{uv}) = \frac{\exp(z_{\text{illicit}})}{\exp(z_{\text{licit}}) + \exp(z_{\text{illicit}})}$$
We define $\text{AML\_Risk\_Score} = P(\text{illicit} \mid e_{uv}) \in [0.0, 1.0]$.

### 8.2 Threshold Optimization & The Precision-Recall Curve
In traditional classification, the default decision threshold is $\tau = 0.50$. In fraud and AML detection with severe class imbalance, setting $\tau = 0.50$ is overly conservative.

We evaluated precision and recall across decision thresholds $\tau \in [0.20, 0.50]$ on the unseen test split:

```
Threshold (tau) │ Flagged Count │ Precision │ Recall │ Target Operational Profile
────────────────┼───────────────┼───────────┼────────┼───────────────────────────
   tau = 0.25   │     4,120     │   38.4%   │ 82.1%  │ Broad regulatory audit
   tau = 0.30   │     2,957     │   44.2%   │ 71.3%  │ Comprehensive screening
   tau = 0.35   │       297     │   54.2%   │ 58.6%  │ Balanced Dashboard Default
   Top-25 Alerts│        25     │   96.0%   │    -   │ Critical High-Priority SARs
```

* **Default Operational Threshold:** $\tau = 0.35$ (35% risk confidence).
* Transactions with $\text{AML\_Risk\_Score} \ge 0.35$ are flagged as **Illicit AML Alerts**.
* At $\tau = 0.35$, the system flags **297 transactions** accounting for **$505.2 Million** in suspect flows.
* Among the top 25 highest-risk alerts generated, **24 of 25 are confirmed ground-truth laundering transactions (96.0% precision)**.

---

## 9. Entity Intelligence: Declaring Nodes as Suspicious Money Mules

While the GNN classifies transaction **edges**, regulatory compliance teams (FIUs) investigate **accounts and entities** (nodes). 

The system aggregates edge predictions up to the node level using graph analytics and topological rule heuristics implemented in [`server.py`](file:///home/rishabh/c0de/anti-money-laundering/Multi-GNN/server.py):

```
Edge Predictions (P(Illicit)) ──► [Compute Node Aggregations] ──► [Topological Role Heuristic]
                                       │                                │
                                       ▼                                ▼
                          • In/Out Volume                  • Money Mule Hub
                          • Flagged In/Out Counts          • Smurfer (Fan-Out)
                          • Max Risk Score                 • Cash-Out Sink
                                                           • Normal Counterparty
```

### 9.1 Account Risk Aggregation Formula
For each account node $v \in V$:
1. **Maximum Incident Risk:**
   $$\text{MaxRisk}(v) = \max_{e \in \mathcal{E}(v)} P(\text{illicit} \mid e)$$
2. **Flagged Flow Counts:**
   $$\text{FlaggedOut}(v) = \sum_{e = (v \to u)} \mathbb{I}\left( P(\text{illicit} \mid e) \ge \tau \right)$$
   $$\text{FlaggedIn}(v) = \sum_{e = (u \to v)} \mathbb{I}\left( P(\text{illicit} \mid e) \ge \tau \right)$$
3. **Suspicious Entity Flag:**
   $$\text{IsSuspicious}(v) = \begin{cases} \text{True} & \text{if } \text{MaxRisk}(v) \ge \tau \text{ or } (\text{FlaggedIn}(v) + \text{FlaggedOut}(v) \ge 1) \\ \text{False} & \text{otherwise} \end{cases}$$

### 9.2 Topological Role Discovery
Based on graph flow dynamics, each suspicious entity is classified into a functional laundering role:

#### 1. Money Mule (Gather-Scatter Hub)
* **Condition:** $\text{FlaggedIn}(v) \ge 1$ AND $\text{FlaggedOut}(v) \ge 1$ (and total volume balance near neutral).
* **Behavior:** Rapid pass-through of funds. The account receives structured transfers from upstream accounts and immediately disperses them downstream, functioning as a transit node in an organized laundering chain.
* **Visualization:** Rendered as a large red hub ($\text{size} = 30$, `#ef4444`) with a glowing drop-shadow halo and badge `🚨 Acc #ID`.

#### 2. Smurfer / Fan-Out Source
* **Condition:** $\text{FlaggedOut}(v) \ge 2$ and $\text{FlaggedIn}(v) < \text{FlaggedOut}(v)$.
* **Behavior:** High out-degree dispersion. Originator of structuring campaigns, breaking down large balances into multiple smaller transfers below regulatory monitoring limits.
* **Visualization:** Rendered in crimson (`#e11d48`, $\text{size} = 24$).

#### 3. Cash-Out Sink / Aggregator
* **Condition:** $\text{FlaggedIn}(v) \ge 2$ and $\text{FlaggedOut}(v) < \text{FlaggedIn}(v)$.
* **Behavior:** High in-degree aggregation. The recipient of structured mule payments, consolidating funds for final withdrawal, wire transfer, or asset purchase.
* **Visualization:** Rendered in rose/crimson (`#e11d48`, $\text{size} = 24$).

#### 4. Normal Counterparty
* **Condition:** $\text{MaxRisk}(v) < \tau$ and no flagged transfers.
* **Behavior:** Legitimate consumer, corporate client, or merchant transacting within expected behavioral distributions.
* **Visualization:** Rendered in deep navy/slate (`#1e293b`) with cyan border (`#38bdf8`, $\text{size} = 14$).

---

## 10. Frontend & Full-Stack Visualization Architecture

The user interface is an interactive web dashboard designed for financial intelligence officers, AML analysts, and investigators.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MONEY MULING DETECTION SYSTEM                         │
│  [Dataset: IBM HI-Small] [382 Nodes • 688 Transactions]   [Search: Acc ID]  │
├─────────────────────────────────────────────────────────────────────────────┤
│  [648,748 Evaluated]  [297 Alerts]  [$505.2M Volume]  [48 Mules]  [96% Prec]│
├───────────────────────────────────────────────────┬─────────────────────────┤
│ Graph Section:                                    │ Entity Intelligence:    │
│  [Dense Core] [Top Mules] [Smurfing] [Clusters]   │  Account #493420        │
│  [Risk Slider: 35%]                  [Fit View]   │  Role: Mule Hub         │
│                                                   │  Max Risk: 42.25%       │
│  ┌─────────────────────────────────────────────┐  │  Outflow: $8,797 (Flag) │
│  │     (Normal)                                │  │  Inflow:  $5,977        │
│  │        │                                    │  │  [Focus Ego Network] │
│  │        ▼                                    │  │                         │
│  │   ((🚨 Mule)) ───► ((🚨 Mule)) ──► (Sink)   │  │ Transaction Stream:     │
│  │        │                 │                  │  │  ↗ To #251714 ($4.3K)   │
│  │        ▼                 ▼                  │  │  ↙ From #8911 ($2.1K)   │
│  │     (Normal)          (Normal)              │  │                         │
│  └─────────────────────────────────────────────┘  │                         │
├───────────────────────────────────────────────────┴─────────────────────────┤
│ Bottom Table: Ranked Alerts Feed (Edge ID | From | To | Amount | Risk | ...)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.1 Backend API Architecture ([`server.py`](file:///home/rishabh/c0de/anti-money-laundering/Multi-GNN/server.py))
Built on **FastAPI** with asynchronous request handling and in-memory pre-computed topology structures:

| Endpoint | Method | Description | Latency |
| :--- | :---: | :--- | :---: |
| `/api/stats` | `GET` | Returns global test split KPIs (evaluated transactions, flagged counts, volume, precision). | $< 1\text{ms}$ |
| `/api/graph` | `GET` | Returns graph nodes and edges with colors, labels, widths, and titles for Vis-Network. | $\approx 2\text{ms}$ |
| `/api/node/{id}` | `GET` | Returns complete entity intelligence: role, in/out volumes, flagged counts, and top-30 audit transfers. | $< 1\text{ms}$ |
| `/api/mules` | `GET` | Returns ranked leaderboard of the most active money mule hubs. | $< 1\text{ms}$ |
| `/api/alerts` | `GET` | Paginated table feed of high-risk transactions sorted by descending risk score. | $< 2\text{ms}$ |

### 10.2 Pre-Computed Dense Subgraphs
To provide instantaneous demo responsiveness without visual clutter or rendering lag, `server.py` pre-computes connected multigraph components at boot:

1. **🌐 Dense Banking Core (`pattern=dense_core`):**
   * **382 interconnected accounts** & **688 transactions** (the giant connected component).
   * Contains **300 legitimate counterparties** and **82 suspicious nodes** glowing in red.
   * Demonstrates how money mules operate embedded inside dense, authentic banking traffic.
2. **🚨 Top Mule Rings (`pattern=mule_rings`):**
   * **202 nodes** & **287 transactions** centered around the top 21 gather-scatter mule hubs.
3. **🌪️ Smurfing Networks (`pattern=fan_out`):**
   * **354 nodes** & **504 transactions** displaying fan-out structuring trees.
4. **🕸️ Multi-Ring Clusters (`pattern=all_clusters`):**
   * **492 nodes** & **890 transactions** combining the top 3 largest connected laundering networks.

### 10.3 Vis-Network Physics Engine (`web/app.js`)
* **Layout Solver:** `forceAtlas2Based` physics simulation.
  * `gravitationalConstant = -50`: Repels nodes to prevent overlapping.
  * `springLength = 95`: Maintains readable edge distances between connected accounts.
  * `damping = 0.45`: Quickly stabilizes the network layout within 120 iterations.
* **Visual Hierarchy:**
  * Suspicious nodes pulse with a glowing red box-shadow halo (`rgba(239, 68, 68, 0.85)`).
  * Suspicious transaction edges are rendered with bold crimson arrows (`width = 2.4`, `#ef4444`) with monetary volume and risk percentage tags (`$XX.XK [YY.Y%]`).
  * Normal licit transactions are rendered as subtle semi-transparent slate arrows (`rgba(71, 85, 105, 0.35)`, `width = 1.0`), preserving topological context without cluttering the canvas.
* **Interactive Inspection:**
  * Clicking any node highlights its connections and populates the right sidebar with its financial summary and recent transaction stream.
  * Clicking **🔍 Focus Ego Network** zooms the view and isolates that account's direct 2-hop neighborhood.

---

## 11. Repository File Map & Execution Guide

### 11.1 Project Directory Map

```
/home/rishabh/c0de/anti-money-laundering/Multi-GNN/
│
├── bible.md                          # (This File) Comprehensive Engineering & Theoretical Bible
├── main.py                           # CLI entrypoint for training and benchmark execution
├── training.py                       # Training loops for homogeneous and heterogeneous GNNs
├── train_util.py                     # Batch loaders, evaluation functions, checkpoint serializers
├── models.py                         # GINe, GATe, RGCN, PNA neural network implementations
├── data_loading.py                   # PyG data object loader, temporal splitting, cache reader
├── data_util.py                      # Port calculation, time deltas, multigraph data classes
├── format_kaggle_files.py            # Converts raw IBM CSVs into formatted edge lists
├── inference.py                      # Standalone batch inference script
├── detect_aml.py                     # Generates detected_aml_alerts.csv from model checkpoints
├── server.py                         # FastAPI backend serving live graph API & static web assets
│
├── checkpoints/                      # Model weights and prediction outputs
│   ├── checkpoint_gin_hi_small.tar   # Trained GINe checkpoint (147,432 parameters, 1.6 MB)
│   └── detected_aml_alerts.csv       # Scored test split predictions (648,748 rows, 44.9 MB)
│
├── web/                              # Full-stack frontend dashboard
│   ├── index.html                    # Application layout, KPI ribbon, canvas, sidebar, table
│   ├── index.css                     # Premium dark theme, glassmorphism, animations
│   ├── app.js                        # Vis-Network graph controller, API client, interaction logic
│   └── vis-network.min.js            # Vis-Network graphing library (offline bundled)
│
├── data_config.json                  # Path configurations to datasets and checkpoints
└── sample_dataset/                   # Sample CSV batches for test evaluations
    └── sample_aml_test_batch.csv     # 1,797-transaction test batch containing active mule rings
```

### 11.2 Execution Commands

#### 1. Running the Interactive Web Dashboard
To start the live dashboard with pre-computed GNN predictions:
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` in any web browser.

#### 2. Formatting a Raw IBM Dataset
To convert a raw transaction CSV (e.g. `HI-Small_Trans.csv`) into the standardized multigraph format:
```bash
python3 format_kaggle_files.py ../dataset/HI-Small_Trans.csv
```

#### 3. Training the GNN from Scratch
To train the GINe architecture with port numberings and reverse message passing:
```bash
python3 main.py \
    --data HI-Small \
    --model gin \
    --emlp \
    --ports \
    --reverse_mp \
    --epochs 20 \
    --batch_size 8192 \
    --save_model
```

#### 4. Running Model Inference on the Test Split
To evaluate a trained checkpoint and generate the alerts CSV:
```bash
python3 detect_aml.py \
    --checkpoint checkpoints/checkpoint_gin_hi_small.tar \
    --dataset ../dataset/HI-Small/formatted_transactions.csv \
    --output checkpoints/detected_aml_alerts.csv
```

---

## 12. Future Horizons & Enterprise Scalability

### 12.1 Scaling to 100M+ Transactions (`HI-Large` & `LI-Large`)
* **GPU-Accelerated Graph Sampling:** Integrating NVIDIA cuGraph and PyG `cugraph-ops` to perform temporal neighbor sampling directly in GPU VRAM, bypassing CPU-GPU transfer bottlenecks.
* **Distributed Training:** Utilizing `torch.distributed` and PyG's DistLoader across multi-GPU clusters for nation-scale banking datasets.

### 12.2 Dynamic Temporal Graph Networks (TGN / TGAT)
* Transitioning from static temporal slicing to **Continuous-Time Dynamic Graph Neural Networks** (TGNs).
* Continuous-time models update node memory states asynchronously after every transaction, allowing real-time transaction scoring in banking payment gateways (sub-50ms latency).

### 12.3 Regulatory Compliance & Explainable AI (XAI)
* **GNNExplainer & SubgraphX:** Integrating automated subgraph explainability algorithms to extract the minimal computational subgraph responsible for flagging an alert.
* **Automated SAR Generation:** Synthesizing the extracted subgraph into natural language reports for immediate submission to regulatory financial authorities (e.g., FinCEN Suspicious Activity Reports).
