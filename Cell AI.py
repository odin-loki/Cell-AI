import numpy as np
import torch
import torch.nn as nn
import ray
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
import time

logging.basicConfig(level=logging.INFO)

@dataclass
class ModelParams:
    """Core system parameters from Section 1.1 of the model"""
    dt: float          # Time step
    D: float          # Diffusion coefficient
    gamma: float      # Decay rate
    eta: float        # Noise amplitude (for η(t))
    num_partitions: int  # Number of parallel partitions
    state_size: int   # Size of state vector per partition


class MemoryCell(nn.Module):
    """
    Implementation of core equations from Section 2.1:
    dS/dt = f(I, S, t) - γS + D∇²S + η(t)
    with parallel decomposition for partitions
    """
    def __init__(self, state_size: int):
        super().__init__()
        # Weight matrices for f(I, S, t) = ∑ᵢ wᵢfᵢ(I, S, t)
        self.W = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        
        # State transition matrix for ΔEij computation
        self.E = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        
        # Activation functions
        self.sigma = nn.Sigmoid()  # For input processing
        self.phi = nn.Tanh()       # For state modulation

    def forward(self, 
               state: torch.Tensor, 
               input_signal: torch.Tensor, 
               neighbor_states: torch.Tensor,
               params: ModelParams) -> torch.Tensor:
        """
        Compute state update according to main equation:
        dS/dt = f(I, S, t) - γS + D∇²S + η(t)
        """
        # 1. Compute f(I, S, t) term
        weighted_input = torch.matmul(self.W, input_signal)
        activation = self.sigma(weighted_input)
        state_coupling = self.phi(torch.matmul(self.E, state))
        f_term = activation * state_coupling

        # 2. Compute diffusion term D∇²S
        if len(neighbor_states) > 0:
            diffusion = params.D * (neighbor_states - state).mean(dim=0)
        else:
            diffusion = torch.zeros_like(state)

        # 3. Compute decay term -γS
        decay = -params.gamma * state

        # 4. Add noise term η(t)
        noise = params.eta * torch.randn_like(state)

        # 5. Compute full state update
        d_state = f_term + diffusion + decay + noise
        
        # 6. Euler integration step
        new_state = state + params.dt * d_state
        
        return new_state


@ray.remote(num_gpus=0.2)  # Assume 5 partitions per GPU
class Partition:
    """
    Implements parallel state evolution for a partition π of the state space
    Following Section 1.1 of the parallel model
    """
    def __init__(self, partition_id: int, params: ModelParams):
        self.id = partition_id
        self.params = params
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory cell and state
        self.cell = MemoryCell(params.state_size).to(self.device)
        self.state = torch.zeros(params.state_size, device=self.device)

    def update(self, 
              input_signal: np.ndarray, 
              neighbor_states: Dict[int, np.ndarray]) -> np.ndarray:
        """Update partition state for one time step"""
        # Convert inputs to tensors
        input_tensor = torch.tensor(input_signal, device=self.device)
        neighbor_tensors = torch.stack([
            torch.tensor(state, device=self.device)
            for state in neighbor_states.values()
        ]) if neighbor_states else torch.empty((0, self.params.state_size), 
                                             device=self.device)

        # Update state
        with torch.no_grad():
            self.state = self.cell(self.state, input_tensor, neighbor_tensors, 
                                 self.params)
        
        # Return state as numpy array
        return self.state.cpu().numpy()

    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.state.cpu().numpy()


class CellularMemorySystem:
    """
    Main system class implementing parallel cellular memory model
    """
    def __init__(self, params: ModelParams):
        ray.init(num_gpus=1)  # Adjust based on available GPUs
        self.params = params
        
        # Initialize partitions
        self.partitions = [
            Partition.remote(i, params) 
            for i in range(params.num_partitions)
        ]

    def _get_neighbors(self, partition_id: int) -> List[int]:
        """Get neighboring partition IDs"""
        neighbors = []
        if partition_id > 0:
            neighbors.append(partition_id - 1)
        if partition_id < self.params.num_partitions - 1:
            neighbors.append(partition_id + 1)
        return neighbors

    async def simulate(self, 
                      input_sequence: List[List[np.ndarray]], 
                      n_steps: int) -> Tuple[Dict[int, List[np.ndarray]], float]:
        """Run parallel simulation"""
        start_time = time.time()
        state_history = {i: [] for i in range(self.params.num_partitions)}

        try:
            for step in range(n_steps):
                # Get current inputs
                current_inputs = (
                    input_sequence[step] 
                    if step < len(input_sequence)
                    else [np.zeros(self.params.state_size) 
                          for _ in range(self.params.num_partitions)]
                )

                # Get all states
                states = await ray.get({
                    i: partition.get_state.remote()
                    for i, partition in enumerate(self.partitions)
                })

                # Update all partitions in parallel
                update_refs = []
                for i, partition in enumerate(self.partitions):
                    # Get neighbor states
                    neighbor_ids = self._get_neighbors(i)
                    neighbor_states = {
                        j: states[j] for j in neighbor_ids
                    }
                    
                    # Update partition
                    update_refs.append(
                        partition.update.remote(current_inputs[i], neighbor_states)
                    )

                # Collect results
                updated_states = await ray.get(update_refs)
                for i, state in enumerate(updated_states):
                    state_history[i].append(state)

                if step % 100 == 0:
                    logging.info(f"Completed step {step}/{n_steps}")

            elapsed_time = time.time() - start_time
            return state_history, elapsed_time

        finally:
            ray.shutdown()


def main():
    # Initialize parameters
    params = ModelParams(
        dt=0.01,          # Time step
        D=0.1,            # Diffusion coefficient
        gamma=0.1,        # Decay rate
        eta=0.01,         # Noise amplitude
        num_partitions=4,  # Number of partitions
        state_size=100    # State vector size
    )

    # Create system
    system = CellularMemorySystem(params)

    # Generate sample input sequence
    n_steps = 1000
    input_sequence = [
        [np.sin(2*np.pi*t*params.dt) * np.ones(params.state_size) 
         for _ in range(params.num_partitions)]
        for t in range(n_steps)
    ]

    # Run simulation
    import asyncio
    state_history, elapsed_time = asyncio.run(
        system.simulate(input_sequence, n_steps)
    )
    
    logging.info(f"Simulation completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()