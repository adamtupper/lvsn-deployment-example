# Deployed jobs on the IID or Christian's Group GPUs

This project serves as an example for deploying training jobs on different servers available at the LVSN.

> **WARNING: I have not modified this example to remove user-specific file paths.** Therefore, you should carefully examine and each file (particularly those listed below) to check for things you will need to personalise. I will create a more generic version of this example in the future.

## This Example

This example shows how to train ResNet-18 models on CIFAR10 with different levels of supervision: 0.08889, 0.25, 0.50, and 1.0.

A labelled proportion of 0.08889 corresponds to 4000 labels (0.08889 * 45,000 examples = 4000.05 labels). This particular baseline is used because it is a popular semi-supervised learning benchmark.

## Organisation

Of particular importance are the following files:

```shell
.dockerignore
Dockerfile
Makefile
requirements.txt
```

They demonstrate how to package your code into a Docker container for deployment. The remaining files simply provide a working example of a basic way to organise your training code in this environment.

## Deployment

1. Copy/clone your code onto a machine.
2. Run `make help` to see the available commands.
3. Start training (`make build` followed by `make run` or `make run-lambda`).

Note that the IID machines run on Lambda Stack, so the `run-lambda` make target should be used. The machines reserved for Christian's group do not, and therefore the `run` target should be used.
