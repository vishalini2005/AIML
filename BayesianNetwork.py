from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
# Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'WetGrass'),
    ('Rain', 'WetGrass')
])
# Define the CPDs (Conditional Probability Distributions)
cpd_cloudy = TabularCPD(
    variable='Cloudy', variable_card=2,
    values=[[0.5], [0.5]]
)
cpd_sprinkler = TabularCPD(
    variable='Sprinkler', variable_card=2,
    values=[[0.5, 0.9],  # P(Sprinkler=0 | Cloudy=0), P(Sprinkler=0 | Cloudy=1)
            [0.5, 0.1]], # P(Sprinkler=1 | Cloudy=0), P(Sprinkler=1 | Cloudy=1)
    evidence=['Cloudy'],
    evidence_card=[2]
)
cpd_rain = TabularCPD(
    variable='Rain', variable_card=2,
    values=[[0.8, 0.2],
            [0.2, 0.8]],
    evidence=['Cloudy'],
    evidence_card=[2]
)
cpd_wetgrass = TabularCPD(
    variable='WetGrass', variable_card=2,
    values=[
        # WetGrass=0
        [1.0, 0.1, 0.1, 0.01],
        # WetGrass=1
        [0.0, 0.9, 0.9, 0.99]
    ],
    evidence=['Sprinkler', 'Rain'],
    evidence_card=[2, 2]
)
# Add CPDs to the model
model.add_cpds(cpd_cloudy, cpd_sprinkler, cpd_rain, cpd_wetgrass)

# Validate the model
assert model.check_model()
# Inference
infer = VariableElimination(model)
# Example Query: What is the probability that WetGrass is wet?
result = infer.query(variables=['WetGrass'])
print(result)