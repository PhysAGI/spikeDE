    
# Explanation of new revision:

## Version 2025-July-10

We have released a new module snn_fx.py which will fully will replace the original snn.py soon.

__TODO__: may fully remove the snn.py file and rename snn_fx.py as snn.py.

The new snn() in snn_fx.py support any flebile call of.

## Before 2025-July

Beside spikeDE, we need to include torchfde:

> pip install git+https://github.com/kangqiyu/torchfde.git

We make some revision for the package to include fractional differential equation solver.

The original code corresponds to odeint_adjoint (default) and odeint with _'euler'_ method, we add fdeint_adjoint and fdeint for fractional differential equation solver. 

For fdeint_adjoint and fdeint, we have the following methods

> fdeint_adjoint:
> - predictor-f
> - predictor-o
> - gl-f
> - gl-o
> - trap-f
> - trap-o

> fdeint:
> - predictor
> - implicitl1 (not suggested)
> - gl
> - trap

```
parser.add_argument('--integrator', type=str, default='odeint',
                        choices=['odeint_adjoint', 'odeint', 'fdeint_adjoint', 'fdeint'],
                        help='differential equation integrator type (default: odeint_adjoint)')

parser.add_argument('--beta', type=float, default=0.5,
                    help='fractional derivative order from 0.0 to 1.0 ( 0 < beta <= 1.0) (default: 0.5)')
    ###only valid when choose fdeint or fdeint_adjoint
    ### if choose odeint or odeint_adjoint, this parameter will be ignored


parser.add_argument('--method', type=str, default='euler', choices=
[
    'euler',  # odeint_adjoint and odeint only support euler method
    'predictor-f', 'predictor-o', 'gl-f', 'gl-o', 'trap-f', 'trap-o',  # fdeint_adjoint method
    'predictor', 'implicitl1', 'gl', 'trap'  # fdeint method
],
                    help='method for euler solver')
parser.add_argument('--step_size', type=float, default=1.0,
                    help='Integration step size (default: 1.0)')
parser.add_argument('--time_interval', type=float, default=1.0,
                        help='Time interval between steps in ms (default: 1.0) for each input event')

```


Furthermore, we need to add ```beta=args.beta``` to finetune the order parameters _beta_ for the fractional differential equation.

Here beta is between 0.0 and 1.0.

```
snn_model = snn(snet, integrator=args.integrator, beta=args.beta).to(device)
```

```
model(data, data_time, output_time=output_time, method=method,
                                                       options=options)
```

Notice:

- For odeint_adjoin and odeint, the beta parameter is invalid. No need to finetune it. 
- For fdeint_adjoint and fdeint, please keep step_size==time_interval. Furthermore, the save_memory is disabled in dvs-xxx files. The parameter output_time will not be used in the solver even we include it to keep code compatibility.

