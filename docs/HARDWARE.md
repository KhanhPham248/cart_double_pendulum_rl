# Plant and policy contracts

The measured MJCF in `assets/cart_double_pendulum_real.xml` defines the plant.
ONNX exporters stamp the policy with a plant contract; loaders validate it before
inference. This prevents applying an export to an incompatible model.

The runtime applies the normalized-action rail guard before conversion to physical
actuator force. Preserve observation order, action scaling, history/recurrent
state and the control period when integrating a policy with another runtime.

Simulation tests, demonstration videos and ONNX contract checks establish separate
properties. This repository does not claim that these artifacts alone qualify a
policy for physical hardware operation.
