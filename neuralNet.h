#ifndef NEURALNET_H
#define NEURALNET_H

#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<cassert>
#include<cmath>
#include"neuron.h"



class Net{
    std::vector<Layer> m_layers; 
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
  public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
};


#endif
