Inherent things that need to be solved with AI:

Over-fitting and imprecision of neural networks is a problem. 

*Note: **precision** is defined as the degree to which the internal values of nodes are precisely tuned all their inputs and expected outputs. This is important to correct functioning of the new neural network*

Neural networks have either a set number of nodes and layers or they have some approximate method of rescaling the network. However, condensing a set of nodes into a singular node that preforms a function would reduce computing power and result in a more precise representation of what the neural network is trying to preform. The precision resulting from condensing will yield to more accurate changes in the neural network.

An additional problem that may arise is that a relatively precise but inaccurate (probably from a small training set at the point in time) general solution may be reached; However, to reach greater accuracy, there may need to be an additional calculations rather than fine tuning existing calculations. This would be difficult with a pre-exiting set of nodes that already are tuned for the generalized solution. 

Sigmoids to a compactified sigmoid is impossible. Thus, arithmetic functions will be favored. However, there are cases where a sigmoid would be what the neural network is trying to represent. Indeed, many cases exist where higher-order mathematical concepts is what neural networks are trying to represent. Due to the previous reasons, non-arithmetic functions will be possible. There will also be instances where the neural network will realize that it needs an additional node, and that the node would have to fulfill a purpose.

With these changes -- specifically the increased precision, there is an increased chance of being able to 'solve' the neural network for a set of inputs. This is in contrast to different solutions depending on what sequence the training set is fed to it.

It is hard to figure out how to compactify and expand the neural nodes. For condensing it could start out with sigmoids, but recognize multiplicative, subtractive, additive, derivative, compounding, etc patterns. For expansion, there would likely need to be value(s) applied to the node, perhaps even just 'inaccuracy', that will spawn an additional sigmoid that probably will then be compactified in a different way. In either case, a robust description of the interaction and differences of the functions the nodes is required. Additionally, it seems that the functions require at least 2 and possibly 9, but probably 4 variables that track inaccuracy.

There will be two parts to back propagation. The first part is where existing values are changed by finding the average of all expected outputs and changing the function to fit the average. The second part is calculating a rolling inaccuracy. Using a rolling average is only valid if no downstream or upstream nodes changes, so the neural network will have to recalculate previous values for nodes upstream and downstream. This can be done by storing the node's inputs each run. However, this could saturate the memory bandwidth (3.8GHz\*4cores : 2xDDR400 $\approx$ 2.4:1. Assuming 2 bytes), 4.8 : 1) and increase processing power. The best way of doing this is to have an estimated memory bandwidth, estimated CPU bandwidth, and estimated clocks per node, then store only some percentage of inputs. There is probably a way to mathematically simplify the process of recalculating (like $F(\bar I) = \overline{^EO}$, and in this case there would only one value stored), and in specific way to maximize computational resources (like doing it in batches to increase register and cache efficiency).

If there is addition/subtraction that is inaccurate, and the expected output would be something like this: (where green is pure addition, black in the extrapolated expected like, and purple are the expected outputs of the runs. x=3 is the average (where the lines intersect))
![[2023-01-31 23_21_11-Desmos _ Graphing Calculator.png | 200]] 
$\text {Standard Deviation of } O_N = \sigma (\mathbb{O_N})$  
$(^\downarrow \alpha, ^\uparrow \beta)(I + X_0)$ $\rightarrow$ 
[
$\mathbb I^- = \forall \mathbb I < \bar I$
$\mathbb I^+ = \forall \mathbb I > \bar I$
$\text {Upper Slope: } ^US = [\sum_{I=0}^{\mathbb I^+}[\frac{^EO_N(I) - (n + X)}{^EO_N(I)}]$
$\text {Lower Slope: } ^LS = [\sum_{I=0}^{\mathbb I^-}\frac{[^EO_N(I) - (I + X)}{^EO_N(I)}]$
$^US < 0 \,\, \& \,\, ^LS > 0 \rightarrow$ $\bar I \times X_1 = \overline{^EO}$

Sigmoid functions:
As seen in $I + X \rightarrow ^RS$, it is impractical to use a conventional sigmoid. Instead, something like $(\text{sign})\lvert1-\frac{X_1}{I + X_1}\rvert$ or a natural log may be used. Either one increases imprecision and processing power, but division should be less.

## Relationships:
---
$^RS$ = real sigmoid, $S$ = artificial sigmoid: $(\text{sign})\lvert1-\frac{X_1}{I + X_1}\rvert$,  $\alpha$ = accuracy, $\beta$ = precision, N = node, n = a number (frequently a given run of the network). Precision of a calculation is defined as how close the average output is to the expected output ($\beta = \overline{^E O} - F(\bar I)$ ). The inaccuracy ($\alpha^-$) of nodes is defined as how much a given node's input changes from the average, assuming high precision, relative to how much the end result is inaccurate ( $\alpha^- = (\sum_n^{\mathbb n}\lvert\frac{(\bar I-I_n) /\bar I}{^NO / ^EO}\rvert) \, \, / \, TotalNum$ ).

### Expansion:
---
#### Elementary Operations
This requires the tracking of at least two variables, and the equations will change depending on which ones are stored. I assume the easiest by only looking at values $>\bar I$, and the variables being 1. the average distance from $\bar I$, 2. the average deviation from $^EO_N$ $\times$ distance from $\bar I$, and 3.  $\bar I$. This is probably the most variable efficient way of doing this without having to recalculate values. The problem is whenever

$I + X\rightarrow$ $\times$: 
-  $\text {Upper Slope: } ^US = [\sum_{I=0}^{\mathbb I^+}[\frac{^EO_N(I) - (n + X)}{^EO_N(I)}]$
- $\text {Lower Slope: } ^LS = [\sum_{I=0}^{\mathbb I^-}\frac{[^EO_N(I) - (I + X)}{^EO_N(I)}]$
- $^US < 0 \,\, \& \,\, ^LS > 0 \rightarrow$ $\bar I \times X_1 = \overline{^EO}$
$I \times X_0 \rightarrow +$: 
-  $\text {Upper Slope: } ^US = [\sum_{I=0}^{\mathbb I^+}[\frac{^EO_N(I) - (n + X)}{^EO_N(I)}]$
- $\text {Lower Slope: } ^LS = [\sum_{I=0}^{\mathbb I^-}\frac{[^EO_N(I) - (I + X)}{^EO_N(I)}]$
- $^US > 0 \,\, \& \,\, ^LS < 0 \rightarrow$ $\bar I + X_1 = \overline{^EO}$

#### Complex Operations
I think that these can use the same variables as elementary operations.

$I + X \rightarrow ^RS$
- $\sum_I^{\mathbb I}[(1+e^{\frac{\ln(^EO^{-1}-1)}{\bar I + X_0} \times (I + X_0)})^{-1} + X_0 - ^EO < ^EO - I - X \rightarrow$
- $\frac{\ln (^EO^{-1} - 1)}{\bar I + X_0} = X_1 \, ;\, (1+e^{X_1\times I})^{-1}$
	- negative is accounted for.

$I + X \rightarrow S$
- Assuming high precision and multiplication as an alternative has been accounted for,
- $^\overline{EO_N(\mathbb I^+)} < 
- $X_1 = 1+\bar I - \overline{I^+}$




Bell curve:  

Equation for k given n amount of points:

m;

For (n; n < points; n++) { m += k[n] * ( (h[n-1]-h[n]) / (k[n]) - k[n-1]) * (k[n] / 2 - k[n-1] + h[n-1])}

h = (y - m) / 2 

k = \frac{hk_{0}-m+y+\sqrt{y\left(y+2\left(2hk_{0}-m\right)\right)+k_{0}\left(6h_{0}\left(m-y\right)+h\left(hk_{0}-4m\right)\right)+m^{2}}}{3h_{0}-h}

  

 Equation for x given n amount of points:

m;

for (int n; n < POINTS; n++) { m += \left(k-k_{0}\right)\left(\frac{h-h_{0}}{2}+h_{0}\right) }

  

output = m + \left(x-k_{0}\right)\left(\frac{\left(x-k_{0}\right)\left(-h_{0}+h\right)}{2\left(k-k_{0}\right)}+h_{0}\right)