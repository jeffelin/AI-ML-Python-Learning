
class Value: 

    def __init__(self, data, _children=(), _op='', label = ''): # self is a requirement for an inwards pass 
        self.data = data 
        self.grad = 0.0 
        self._backward = lambda: None 
        self._prev = set(_children) # set and convenience
        self._op = _op  
        self.label = label

    def __repr__(self): # official string 
        return f"Value(data = {self.data})" # repr function to return a string 
    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += 1.0*out.grad 
            other.grad += 1.0*out.grad 
        out._backward = _backward 

        return out 
    def __neg__(self): 
        return self * -1 
    def __sub__(self, other): 
        return self + (-other)    
      
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')
        def _backward():
            self.grad = other.data*out.grad 
            other.grad = self.data*out.grad 
        out._backward = _backward 
        return out 
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.grad*out.grad  
        out._backward = _backward 
        return out 
        
    
    def __rmul__(self, other): 
        return self * other 

    def __truediv__(self, other): 
        return self * other**-1    


    def __pow__(self, other): 
        assert isinstance (other, (int, float))
        out = Value (self.data**other, (self,), f'**(other)')   
        def backward(): 
            self.grad += other * self.data ** (other-1) * out.grad 
        out._backward = backward() 

        return out 


    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad = (1 - t**2 ) * out.grad 
        out._backward = _backward 

        return out
    
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

from graphviz import Digraph 

def trace(root): 
# builds a set of all nodges and edges in a graph
    nodes, edges = set(), set() 
    def build(v): 
        if v not in nodes: 
            nodes.add(v)
            for child in v._prev: 
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges 

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'}) # left to right 

    nodes, edges = trace(root) 
    for n in nodes: 
        uld = str(id(n))

        dot.node(name = uld, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data,n.grad), shape = 'record')
        if n._op:
            dot.node(name = uld + n._op, label = n._op)

            dot.edge(uld+n._op, uld)

    for n1, n2 in edges: 
        dot.edge(str(id(n1)), str(id(n2)) +n2._op)
    return dot



class Neuron: 
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1)) 
    
    def __call__(self, x): 
        
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) 
        out = act.tanh() 
        return out  
# nin is number of inputs, nouts is number of neurons in a single layer 
    def parameters(self): 
        return self.w + [self.b]
class Layer: 
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x): 
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): 
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP: 
    def __init__(self, nin, nouts):
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x): 
        for layer in self.layers:
            x = layer(x) 
        return x  
    def parameters(self): 
        return [p in layer for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]  
n = MLP(3, [4, 4, 1]) 

xs = [ 
    [2.0, 3.0, -1.0], 
    [3.0,-1.0, 0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets, appears to be binary classification 


#Calculating the loss function to help tune or measure the neural net to the intended targets 
#We want to minimize the loss 
ypred

for k in range(10): 
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    #backward passto calculate the proper gradients "learning phase "
    for p in n.parameters(): 
        p.grad = 0.0 
    loss.backward()
    #update
    for p in n.parameters():
        a = -0.01 
       
        p.data += a*p.grad 

    print(k, loss.data)

    

