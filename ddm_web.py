import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

class DDMSimulator:
    def __init__(self):
        self.setup_params()
        
    def setup_params(self):
        st.title("Drift Diffusion Model Simulator")
        
        # Create three columns for parameters, equations, and description
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Model Parameters")
            self.params = {
                'drift_rate': st.slider('Drift Rate (v)', -3.0, 3.0, 1.5, 
                                      help="Average rate of evidence accumulation (higher = faster decisions)"),
                'threshold': st.slider('Threshold (a)', 0.5, 5.0, 2.0,
                                     help="Decision boundary (higher = more accurate but slower)"),
                'bias': st.slider('Starting Bias (z)', -2.0, 2.0, 0.0,
                                help="Starting point of evidence (positive = bias toward upper boundary)"),
                'noise_sd': st.slider('Noise (σ)', 0.1, 3.0, 1.0,
                                    help="Standard deviation of noise in evidence accumulation"),
                'dt': 0.05
            }
            
        with col2:
            st.subheader("Model Equations")
            st.markdown("""
            **Evidence Accumulation:**
            
            $dE = v \cdot dt + σ \sqrt{dt} \cdot N(0,1)$
            
            where:
            - $E$ = Evidence
            - $v$ = Drift rate
            - $σ$ = Noise level
            - $dt$ = Time step
            - $N(0,1)$ = Standard normal noise
            
            **Decision Rule:**
            
            $|E| ≥ a$ = Decision made
            
            where:
            - $a$ = Threshold
            
            **Initial Condition:**
            
            $E(0) = z$
            
            where:
            - $z$ = Starting bias
            """)

        # Initialize state if not exists
        if 'time' not in st.session_state:
            st.session_state.time = 0
            st.session_state.evidence = self.params['bias']
            st.session_state.evidence_history = [self.params['bias']]
            st.session_state.time_history = [0]
            st.session_state.decision_made = False
            st.session_state.running = False
            
        with col3:
            st.subheader("Model Description")
            st.markdown("""
            This simulator shows how decisions might be made in the brain using a 
            Drift Diffusion Model (DDM).
            
            **Parameters Explained:**
            - **Drift Rate (v)**: Average speed of evidence accumulation
                - Positive = bias toward upper boundary
                - Negative = bias toward lower boundary
            
            - **Threshold (a)**: How much evidence needed for decision
                - Higher = more accurate but slower
                - Lower = faster but more error-prone
            
            - **Starting Bias (z)**: Initial evidence level
                - Positive = head start toward upper boundary
                - Negative = head start toward lower boundary
            
            - **Noise (σ)**: Random fluctuations in evidence
                - Higher = more random decisions
                - Lower = more deterministic
            """)

    def update_simulation(self):
        if not st.session_state.decision_made and st.session_state.running:
            # Update evidence multiple steps at once for speed
            for _ in range(5):  # Process 5 steps at once
                noise = self.params['noise_sd'] * np.sqrt(self.params['dt']) * np.random.normal()
                new_evidence = st.session_state.evidence + (self.params['drift_rate'] * self.params['dt'] + noise)
                
                # Update histories
                st.session_state.time += self.params['dt']
                st.session_state.evidence = new_evidence
                st.session_state.evidence_history.append(new_evidence)
                st.session_state.time_history.append(st.session_state.time)
                
                # Check for decision
                if abs(new_evidence) >= self.params['threshold']:
                    st.session_state.decision_made = True
                    decision_boundary = "Upper" if new_evidence >= self.params['threshold'] else "Lower"
                    st.success(f"Decision made: {decision_boundary} boundary crossed at {st.session_state.time:.2f} seconds")
                    break

    def plot_trial(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot current trajectory
        ax.plot(st.session_state.time_history, st.session_state.evidence_history, 'k-', label='Evidence', linewidth=2)
        ax.axhline(y=self.params['threshold'], color='r', linestyle='-', label='Upper Threshold')
        ax.axhline(y=-self.params['threshold'], color='b', linestyle='-', label='Lower Threshold')
        ax.plot(0, self.params['bias'], 'go', markersize=10, label='Starting Point')
        
        ax.set_xlim(0, 5)  # Fixed time window
        ax.set_ylim(-self.params['threshold']*1.5, self.params['threshold']*1.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Evidence')
        ax.set_title('Drift Diffusion Model Simulation')
        ax.legend()
        ax.grid(True)
        
        return fig

def main():
    st.set_page_config(page_title="DDM Simulator", layout="wide")
    
    simulator = DDMSimulator()
    
    # Control buttons in a row
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button('Start' if not st.session_state.running else 'Pause'):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button('Reset'):
            st.session_state.time = 0
            st.session_state.evidence = simulator.params['bias']
            st.session_state.evidence_history = [simulator.params['bias']]
            st.session_state.time_history = [0]
            st.session_state.decision_made = False
            st.session_state.running = False
    
    # Create placeholder for plot
    plot_placeholder = st.empty()
    
    # Main simulation loop
    while True:
        simulator.update_simulation()
        plot_placeholder.pyplot(simulator.plot_trial())
        time.sleep(0.001)  # Reduced sleep time for faster updates
        
        if not st.session_state.running:
            break
        
        if st.session_state.decision_made:
            st.session_state.running = False
            break

if __name__ == "__main__":
    main() 