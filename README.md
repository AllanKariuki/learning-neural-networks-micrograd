# Micrograd From Scratch

A small scalar-valued autograd engine built step by step in a Jupyter notebook (`micrograd_from_scratch_tt.ipynb`), following the ideas behind Andrej Karpathy's *micrograd*. It builds a computation graph out of `Value` nodes, runs a forward pass, and then uses **backpropagation** (the chain rule applied over a topologically sorted graph) to compute the gradient of the output with respect to every node.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyterlab graphviz
jupyter lab
```

`graphviz` also needs the system binary to render graphs (`brew install graphviz` on macOS).

---

## 1. The `Value` node

Every number in the graph is wrapped in a `Value` object:

| Attribute   | Purpose |
|-------------|---------|
| `data`      | The scalar value computed in the forward pass. |
| `grad`      | ∂L/∂(this node): how much the final output changes if this value is nudged. Starts at `0.0`. |
| `_prev`     | The set of child nodes that produced this value (its inputs). |
| `_op`       | The operation that produced it (`'+'`, `'*'`, `'tanh'`), used for drawing. |
| `_backward` | A closure that pushes this node's gradient down to its children. Leaf nodes keep the default no-op. |
| `label`     | A human-readable name for the graph visualisation. |

Each operation creates a new `Value` for its result, records its inputs in `_prev`, and attaches a `_backward` function that knows the **local derivative** of that operation.

## 2. Operations and their local gradients

The gradient of any node is computed with the **chain rule**:

```
dL/d(input) = dL/d(output) * d(output)/d(input)
              └── out.grad ┘   └── local derivative ┘
```

Each `_backward` closure multiplies the incoming gradient (`out.grad`) by the local derivative and **adds** (`+=`) it to the input's `grad`.

### Addition: `out = a + b`

```python
self.grad  += 1 * out.grad
other.grad += 1 * out.grad
```

d(a+b)/da = 1 and d(a+b)/db = 1, so addition simply **routes** the gradient unchanged to both inputs.

### Multiplication: `out = a * b`

```python
self.grad  += other.data * out.grad
other.grad += self.data  * out.grad
```

d(a·b)/da = b and d(a·b)/db = a, so each input receives the gradient scaled by the **other** input's value.

### Activation: `out = tanh(x)`

```python
t = (e^(2x) - 1) / (e^(2x) + 1)
self.grad += (1 - t**2) * out.grad
```

d tanh(x)/dx = 1 − tanh(x)², which can be computed directly from the output `t` without recomputing anything.

### Why `+=` and not `=`?

If a node is used more than once (e.g. `b = a + a`), gradient flows back to it along several paths. By the multivariable chain rule those contributions must be **summed**. Using `=` would overwrite one path with another and give the wrong answer.

## 3. The backward pass: `backward()`

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

How it works:

1. **Build a topological order.** `build_topo` does a depth-first search from the output. A node is only appended to `topo` *after* all of its children have been appended, so the list is ordered from the inputs (leaves) to the output.
2. **Seed the output gradient.** `self.grad = 1.0` because dL/dL = 1.
3. **Walk the list in reverse.** Going from the output back to the inputs guarantees that when a node's `_backward()` runs, its own `grad` is already complete (every node that depends on it has already pushed its contribution). It then passes that gradient on to its children.

Calling `_backward()` manually in the wrong order (as experimented with in the notebook) would propagate gradients that haven't been fully accumulated yet. The topological sort is what makes the order correct automatically.

## 4. Visualising the graph: `trace` and `draw_dot`

- `trace(root)` walks back from the output and collects every node and every `(child, parent)` edge.
- `draw_dot(root)` uses Graphviz to draw the graph left-to-right. Each `Value` is a record box showing `label | data | grad`, and each operation is a small node (`+`, `*`, `tanh`) sitting between its inputs and its result.

Calling `draw_dot(L)` or `draw_dot(o)` lets you see the forward values and, after backprop, the gradients on every node.

---

## 5. Walkthrough: Example 1 (manual backprop)

```python
a = Value(2.0);  b = Value(-3.0);  c = Value(10.0);  f = Value(-2.0)
e = a * b        # -6
d = e + c        #  4
L = d * f        # -8
```

Gradients computed by hand, working backwards with the chain rule:

| Node | Derivation | Gradient |
|------|-----------|----------|
| L | dL/dL | 1 |
| d | dL/dd = f | -2 |
| f | dL/df = d | 4 |
| c | dL/dc = dL/dd · dd/dc = -2 · 1 | -2 |
| e | dL/de = dL/dd · dd/de = -2 · 1 | -2 |
| a | dL/da = dL/de · de/da = -2 · b = -2 · -3 | 6 |
| b | dL/db = dL/de · de/db = -2 · a = -2 · 2 | -4 |

Notice how the `+` node passes `-2` straight through to both `c` and `e`, and the `*` nodes swap in the other input's value.

The `lol()` cell is a numerical sanity check: it evaluates L twice, nudging one input by a small `h` the second time, and computes `(L2 - L1) / h`. That slope should match the analytical gradient for whichever input was nudged.

## 6. Walkthrough: Example 2 (a single neuron)

A neuron with two inputs, two weights, a bias and a `tanh` activation:

```python
x1 = 2.0,  x2 = 0.0      # inputs
w1 = -3.0, w2 = 1.0      # weights
b  = 6.7                 # bias

n = x1*w1 + x2*w2 + b    # 0.7
o = tanh(n)              # ≈ 0.6044
```

Backprop by hand:

| Node | Derivation | Gradient |
|------|-----------|----------|
| o | do/do | 1 |
| n | 1 − o² | ≈ 0.6347 |
| b, x1w1+x2w2 | `+` routes the gradient | ≈ 0.6347 |
| x1w1, x2w2 | `+` routes the gradient | ≈ 0.6347 |
| x1 | w1 · 0.6347 | ≈ -1.9042 |
| w1 | x1 · 0.6347 | ≈ 1.2695 |
| x2 | w2 · 0.6347 | ≈ 0.6347 |
| w2 | x2 · 0.6347 | 0 |

`w2.grad` is 0 because `x2` is 0: changing `w2` has no effect on the output while its input is zero.

The notebook computes these three ways:

1. **By hand** — setting each `.grad` directly using the formulas above.
2. **Step by step** — setting `o.grad = 1.0` and calling each node's `_backward()` in order from the output inwards.
3. **Automatically** — `o.backward()`, which builds the topological order and does all of the above in one call.

All three produce the same gradients.

## 7. What the gradients mean

A node's `grad` tells you how the final output responds to a small increase in that node:

- **Positive** grad → increasing the value increases the output.
- **Negative** grad → increasing the value decreases the output.
- **Magnitude** → how strongly it influences the output.

This is exactly the signal used in training: nudge each weight a small step in the direction that reduces the loss (gradient descent).
