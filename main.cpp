#include"neuralNet.h" // including the custom neural net library
// the library includes vector iostream and string
//This program is based on the video by David Miller on Vimeo

using namespace std;

int main()
{
  vector<unsigned> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);
  Net myNet(topology);
  vector<double> inputVals;
  myNet.feedForward(inputVals);
  vector<double> targetVals;
  myNet.backProp(targetVals);
  vector<double> resultVals;
  myNet.getResults(resultVals);
}

