from Agent import Agent

if __name__ == '__main__':
    Agent = Agent(Environment="Pong-v0",
                  Load_Weights=False,
                  Weights_Path="recent_weights.hdf5")
    Agent.play()