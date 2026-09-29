# EndAI — Evolving Neural Dynamics in Agent Interaction

EndAI is a neuroevolution project exploring the evolution of neural dynamics in interacting agents.

The project evolves recurrent neural network controllers through evolutionary processes rather than backpropagation. Neuroevolutionary algorithms are often stigmatized for their inefficiency due to the vast search spaces to which brute-force exploration is applied. Therefore, the goal of this project is to experiment with **parameter-efficient evolution** of populations of interacting agents.

The hope is to discover novel adaptive neural dynamics that can arise in relatively unconstrained systems.

At this early stage of the project, the evolutionary algorithms are intentionally simple and rely on brute-force exploration. Agents inhabit a simulated environment, interact with it and with each other, and are evaluated according to their resulting fitness.

## Demo

The best place to start is the [**EndAI demo notebook**](https://github.com/PrimozRavbar/EndAI-evolving-neural-dynamics-in-agent-interaction/blob/main/EndAI_DEMO.ipynb).

The notebook contains an explanation of the project, the simulation setup, and a complete example of loading evolved genomes and running an agent simulation.

## Repository

```text
EndAI-evolving-neural-dynamics-in-agent-interaction/
│
├── endai/                    # EndAI source code
├── genomes/                  # Genome pools and example genomes
│   └── genomes_examples/
├── EndAI_DEMO.ipynb          # Project explanation and runnable demo
└── README.md
```

## Status

EndAI is an active research and development project.

The code and evolutionary experiments are being developed incrementally.

## License

MIT

