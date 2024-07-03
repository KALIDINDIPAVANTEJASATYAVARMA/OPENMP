#include <atomic>
#include<vector>
#include<algorithm>
#include "../../atomicUtil.h"
#include "../../graph.hpp"

void Compute_SSSP(graph& g,int* weight,int* dist,int src)
{

   bool* modified=new bool[g.num_nodes()];
   bool* modified_nxt=new bool[g.num_nodes()];
   #pragma omp parallelfor
   for (int t = 0; t<g.num_nodes(); t ++) 
    {
    dist[t] = INT_MAX;
    modified_nxt[t]=false;
    modified[t] = false;    
    }


  modified[src] = true;
  dist[src] = 0;
 
  bool finished = false;
 
  while ( !finished )
  {
    finished=true;
    #pragma omp parallel
   {
    #pragma omp for
    for (int v = 0; v < g.num_nodes(); v ++) 
    { 

      if (modified[v] == true )
        {
        for (int edge = g.indexofNodes[v]; edge < g.indexofNodes[v+1]; edge ++) 
        {  
           int nbr = g.edgeList[edge] ;
           int e = edge;
           int dist_new = dist[v] + weight[e];
        
            bool modified_new = true;
            if(dist[nbr]>dist_new)
               {
                 int oldValue = dist[nbr];
                 atomicMin(&dist[nbr],dist_new);
                 finished=false;  
                 modified_nxt[nbr]=modified_new;
              }

          }
      }
     
    }

    #pragma omp for
    for (int v = 0; v <g.num_nodes(); v ++) 
      {
         modified[v]=modified_nxt[v];
         modified_nxt[v]=false;
      }  
   }
  
  }  
}

int main()
{ 

 
   graph G("../../dataRecords/Email-Enron.txt");
   G.parseGraph();

  int* edgeLen=G.getEdgeLen();
  int* dist=new int[G.num_nodes()+1];

   int src=0;
   double startTime=omp_get_wtime();
   Compute_SSSP(G,edgeLen,dist,src);
   double endTime=omp_get_wtime();
   printf("RunTime : %f\n",endTime-startTime);

   for (int i = 0; i <10; i++)
  {
    printf( "%d  %d\n", i, dist[i]);
  }

  return 0;
}
