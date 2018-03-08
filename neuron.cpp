#include"neuron.h"


// Connection Struct constructor
// Maybe create a new header for it?

Connection::Connection(){
  weight = (double)rand()/(double)RAND_MAX;
  deltaWeight = (double)rand()/(double)RAND_MAX;
}



// Class Neurons functions


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
  for(unsigned c = 0; c < numOutputs; ++c){
    m_outputWeights.push_back(Connection());

  }
  m_myIndex = myIndex;
  eta = 0.15;
  alpha = 0.5;
}

void Neuron::feedForward(const Layer &prevLayer){
  double sum = 0.0;

  for(unsigned neuron = 0; neuron < prevLayer.size(); ++neuron){
    sum += prevLayer[neuron].getOutputVals() *
        prevLayer[neuron].m_outputWeights[m_myIndex].weight;
  }
  m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x){
  // tanh(x) { range is between -1.0 to 1.0
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
  return 1 - x*x; // quick approximation of d(tanh(x))
  // actual derivative is d(tanh(x))/dx = 1 - ( tanh(x) * tanh(x) )
}


void Neuron::setOutputVals(double value){
  m_outputVal = value;
}

double Neuron::getOutputVals() const{
  return m_outputVal;
}

void Neuron::calcOutputGradients(double targetVal){
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const{
  double sum = 0.0;

  for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }
  return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer){
  for(unsigned n = 0; n < prevLayer.size(); ++n){
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight =
          eta * neuron.getOutputVals() * m_gradient + alpha * oldDeltaWeight;
    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}



