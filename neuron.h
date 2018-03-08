#ifndef NEURON_H
#define NEURON_H

#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<cassert>
#include<cmath>

class Neuron;

typedef std::vector<Neuron> Layer; 


struct Connection{
  double weight;
  double deltaWeight;
  Connection();
};

//************Class Neuron***********

class Neuron{
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double eta;
    double alpha;

  public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer &prevLayer);
    void setOutputVals(double value);
    double getOutputVals() const;
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    double sumDOW(const Layer &nextLayer) const;
    void updateInputWeights(Layer &prevLayer);
    double m_gradient;

};
//************Class Net**************


#endif
