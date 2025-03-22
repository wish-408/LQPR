# Explanation of Pattern Markings
Given an interval $[v_i, v_{i + 1}]$ over the value $v$ of a performance metric $y$, the fragment and its implied preference of the requirement, denoted as $\psi$, can be represented as Backus-Naur notations in `LQPR` below:

$$\psi ::=  \mathcal{G} \mid \mathcal{S} \mid \mathcal{E}$$

where 

$$\mathcal{G} ::= \forall v \in [v_i,v_{i+1}], \text{ a greater } v \text{ is preferred at } [s_i,s_{i+1}]$$

$$\mathcal{S} ::= \forall v \in [v_i,v_{i+1}], \text{ a smaller } v \text{ is preferred at } [s_i,s_{i+1}]$$

$$\mathcal{E} ::= \forall v \in [v_i,v_{i+1}] \text{ is equally preferred at } s_i$$

where $s_i$ denotes the satisfaction score for that interval ($s_i \in [0,1]$), which is adapted depending on the preference of the adjacent intervals in a performance requirement.


In `patterns.txt`, you will see the following content:
```
100 percent of $1 0
within 100 $0 -1
later 100 $1 0
decreased by 100 $0 -1
out of 100 $1 0
longer than 100 $1 0
more than 100 $1 0
in under 100 $0 -1
less than 100 $0 -1
a period of 100 $1 -1
......
```
Each line is a $pattern \rightarrow label$ combination, separated by the symbol `$`. 
The label is composed of a binary tuple. The numeric symbol `1` represents the symbol $\mathcal{G}$ mentioned in the above text, `−1` represents the symbol $\mathcal{S}$, and `0` represents the symbol $\mathcal{E}$. 
