from pandas import DataFrame, read_csv
from pandas import ExcelWriter
from folium.features import DivIcon
from collections import defaultdict
from pandas.io.json import json_normalize
import sys            
import csv
import folium        
import urllib2
import json
import pandas as pd
import webbrowser 
import pandas as pd
import numpy as np
import copy



mapquestKey = 'tKFF3XDTAOzJ8GYjUtVkIQKkCrbuUsXj'

# To get shape points of s-t quasi pat
def genShapepoints(startCoords, endCoords):
    # We'll use MapQuest to calculate.
    transportMode = 'fastest'        # Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)

    # assemble query URL
    myUrl = 'http://www.mapquestapi.com/directions/v2/route?key={}&routeType={}&from={}&to={}'.format(mapquestKey, transportMode, startCoords, endCoords)
    myUrl += '&doReverseGeocode=false&fullShape=true'

    # print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
    # print myUrl

    # Now, we'll let Python go to mapquest and request the distance matrix data:
    request = urllib2.Request(myUrl)
    response = urllib2.urlopen(request)
    data = json.loads(response.read())

    # print "\nHere's what MapQuest is giving us:"
    # print data

    # retrieve info for each leg: start location, length, and time duration
    lats = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lat'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
    lngs = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lng'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
    secs = [data['route']['legs'][0]['maneuvers'][i]['time'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
    dist = [data['route']['legs'][0]['maneuvers'][i]['distance'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]

    # print "\nHere are all of the lat coordinates:"
    # print lats

    # create list of dictionaries (one dictionary per leg) with the following keys: "waypoint", "time", "distance"
    legs = [dict(waypoint = (lats[i],lngs[i]), time = secs[i], distance = dist[i]) for i in range(0,len(lats))]

    # create list of waypoints (waypoints define legs)
    wayPoints = [legs[i]['waypoint'] for i in range(0,len(legs))]
    # print wayPoints

    # get shape points (each leg has multiple shapepoints)
    shapePts = [tuple(data['route']['shape']['shapePoints'][i:i+2]) for i in range(0,len(data['route']['shape']['shapePoints']),2)]
    # print shapePts

    return shapePts

# To get the graph netwrok of nodes and edges
class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(list)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)
            self.add(node2, node1)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].append(node2)
        if not self._directed:
            self._graph[node2].append(node1)


    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

# To find a capacity of each edges
def getCapacity(velocity):
    # Spline function to calcuate safe distance
    c = ((velocity*1000/3600)**2)/9.88
    # Add the average length of a car as 5 meters
    d = 1000/(c+5)
    capacity =d*velocity
    return capacity


# to get time for each edges
def getTime(velocity, distance):
    time = (distance/velocity)*60
    return time

# Makes adjacency matrix considering the direction of flow between adjacent nodes
def adjMatrix(capacitydf, flowdf):

    adjMatrixdf = pd.DataFrame(0, index=np.arange(1, len(capacitydf)+1), columns=np.arange(1, len(capacitydf)+1))
    for i in capacitydf:
        for j in capacitydf:
            if capacitydf.loc[i][j] > 0 and capacitydf.loc[j][i] > 0:
                if flowdf.loc[i][j] == 0 and flowdf.loc[j][i] == 0:
                    adjMatrixdf.loc[i][j] = 3
                    adjMatrixdf.loc[j][i] = 3
                elif flowdf.loc[i][j] > 0:
                    adjMatrixdf.loc[i][j] = 1
                    adjMatrixdf.loc[j][i] = 2
                    if flowdf.loc[i][j] == capacitydf.loc[i][j]:
                        adjMatrixdf.loc[i][j] = 0
                        
                elif flowdf.loc[j][i] > 0:
                    adjMatrixdf.loc[j][i] = 1
                    adjMatrixdf.loc[i][j] = 2
                    if flowdf.loc[j][i] == capacitydf.loc[j][i]:
                        adjMatrixdf.loc[j][i] = 0
                
            elif capacitydf.loc[i][j] > 0 and capacitydf.loc[j][i] == 0:
                adjMatrixdf.loc[i][j] = 1
                if flowdf.loc[i][j] > 0:
                    adjMatrixdf.loc[j][i] = 2
                    if flowdf.loc[i][j] == capacitydf.loc[i][j]:
                        adjMatrixdf.loc[i][j] = 0
            
            elif capacitydf.loc[j][i] > 0 and capacitydf.loc[i][j] == 0:
                adjMatrixdf.loc[j][i] = 1
                if flowdf.loc[j][i] > 0:
                    adjMatrixdf.loc[i][j] = 2
                    if flowdf.loc[j][i] == capacitydf.loc[j][i]:
                        adjMatrixdf.loc[j][i] = 0    
                    
                    
    return adjMatrixdf    


# Finds the s-t quasi path
def bfs(adjMatrixdf, s, t):
       
    i = copy.copy(s)
    new = [i, 0, 0]
    lab = pd.DataFrame(data = [new])
    label = 1
    lastLabel = 0
    sDummy = copy.copy(s)
    nodeList = [sDummy]
    elseCounter = 0
    if elseCounter == 0:
        while t not in nodeList and (elseCounter == 0):
            for i in adjMatrixdf:
                if i == lab[0][lastLabel]:
                    for j in adjMatrixdf:
                        if adjMatrixdf.loc[i][j] and j not in nodeList:
                            lab.loc[label] = 0
                            lab[0][label] = j
                            lab[1][label] = sDummy
                            lab[2][label] = adjMatrixdf.loc[i][j]
                            nodeList.append(j)
                            if j != t:
                                label = label + 1
                            else:
                                sDummy = t
            lastLabel = lastLabel + 1
            if lastLabel < lab.shape[0]:
                sDummy = lab[0][lastLabel]
            else:
                elseCounter = 1
            if t in nodeList:
                break
    if nodeList[len(nodeList)-1] == t:
        pathExist = 1
        p = [t]
        pcounter = 0
        while p[pcounter] != s:         
            for i in lab.index:
                if (lab[0][i] == p[pcounter]):  
                    p.append(lab.loc[i][1]) 
                    pcounter += 1
                   
        p.reverse()
        return [p, pathExist, lab]
    else:  
        pathExist = 0
        tempP = []
        for c in lab[0]:
            tempP.append(c)
        p = tempP
        return [p, pathExist, lab]

# Finds the slack    
def fAugmenting(capacitydf, flowdf , p, lab):
    tempSlacks = []
    for k in range(0, len(p)-1):
        for l in lab[0]:
            if (l == p[k+1]):
                tempIndex = lab[0][lab[0] == l].index.tolist()
                if (lab.loc[tempIndex[0]][2] == 3 or (lab.loc[tempIndex[0]][2] == 1)): 
                    tempSlacks.append(capacitydf.loc[p[k]][p[k+1]] - flowdf.loc[p[k]][p[k+1]])
                elif (lab.loc[tempIndex[0]][2] == 2):
                    tempSlacks.append((flowdf.loc[p[k+1]][p[k]]*(-1)))   
    slack = min(abs(x) for x in tempSlacks)
    
    return slack

# Updates the flow
def updateFlow(adjMatrixdf, fIN, p, slack, pathExist):
    if (pathExist ==  1):
        newFlow = fIN
        for k in range(0, len(p)-1):
            #print p[k]
            if (adjMatrixdf.loc[p[k]][p[k+1]] == 1 or adjMatrixdf.loc[p[k]][p[k+1]] == 3):
                newFlow.loc[p[k]][p[k+1]] = fIN.loc[p[k]][p[k+1]] + slack 
            elif (adjMatrixdf.loc[p[k]][p[k+1]] == 2):
                newFlow.loc[p[k+1]][p[k]] = fIN.loc[p[k+1]][p[k]] - slack
            else:
                "ERROR: path has zero adjacancy for edge from %d to %d" % (p(k),p(k+1))
        
        return newFlow
    else:
        return fIN

# Calculates the total time for s-t quasi path
def total_time(tour, trafficInfoDict):
    totalTime = 0
    for i in range(0, len(tour)-1):
        if tour[i] > tour[i+1]:
            for j in trafficInfoDict['edgeID']:
                if tour[i] == trafficInfoDict['endNode'][j] and tour[i+1] == trafficInfoDict['startNode'][j]:
                    totalTime += trafficInfoDict['backwardTime'][j]
        else:
            for j in trafficInfoDict['edgeID']:
                if tour[i] == trafficInfoDict['startNode'][j] and tour[i+1] == trafficInfoDict['endNode'][j]:
                    totalTime += trafficInfoDict['forwardTime'][j]
    return totalTime


# Finds the minimum cut
def mincut(allNodes, p, newCap, newFlow, sources, targets, s, t):
    
    vS = copy.copy(p)
    if len(sources) > 1:
        if t in vS:
            vS.remove(t)
   
    if len(targets) > 1:
        if s in vS:
            vS.remove(s)
            
    vT = []
    for m in allNodes:
        if m not in vS:
            vT.append(m)
   
    minCuts = [] 
    minCutDict = {}
    minCut = 0       
    for k in vT:
        for l in vS:
            if newFlow.loc[l][k] > 0:
                if newCap.loc[l][k] - newFlow.loc[l][k] == 0:
                    tempCuts = []
                    tempCuts.append(l)
                    tempCuts.append(k)
                    c = tuple(tempCuts)
                    minCuts.append(c)
                    minCutDict[c] = newFlow.loc[l][k]
                    minCut += newFlow.loc[l][k]
                    
    return minCuts, minCutDict, minCut


# Finds maxflow, mincut, mincuts, slacks, s-t quasi path using above mentionded function
def fordFulkersion(newCap, newFlow, adjMatrixdf, s, t):
    pathDict = {}
    pathExist = 1
    # pathEcist = 1, if s-t quasi path exists, else 0
    maxFlow = 0 
    pathId = 1
    while pathExist:
        newAdj = adjMatrix(newCap, newFlow)
        [p, pathExist, lab] = bfs(newAdj, s, t)

        slack = fAugmenting(newCap, newFlow, p, lab)
        maxFlow = maxFlow + slack
        fIN = newFlow[:]    
        newFlow = updateFlow(newAdj, fIN, p, slack, pathExist)
        pathDict[pathId] = p
        pathId += 1 
        
    [minCuts, minCutDict, minCut] = mincut(allNodes, p, newCap, newFlow, sources, targets, s, t)
        
    return [newAdj, p, lab, maxFlow, newFlow, minCuts, minCutDict, minCut, pathDict]

# Open the csv file which has all required data    
csvFile = '%s.csv' % sys.argv[1]
df = pd.read_csv(csvFile)

# Add required fields to the data frame
df['forwardCap'] = getCapacity(df['forwardSpeed'])
df['backwardCap'] = getCapacity(df['backwardSpeed'])
df['forwardTime'] = getTime(df['forwardSpeed'], df['Length'])
df['backwardTime'] = getTime(df['backwardSpeed'], df['Length'])
df.forwardCap = df.forwardCap.round()
df.backwardCap = df.backwardCap.round()
trafficInfoDict = df.to_dict()


# Make a dicitonaries and data frames which can be converted into matrix for further calculations

forwardCapDict = {}
forwardCapDict = defaultdict(dict)
backwardCapDict = {}
backwardCapDict = defaultdict(dict)
distanceDict = {}
distanceDict = defaultdict(dict)
forwardSpeedDict = {}
forwardSpeedDict = defaultdict(dict)
backwardSpeedDict = {}
backwardSpeedDict = defaultdict(dict)


for i in range(0, len(df)):

    forwardCapDict[df['startNode'][i]][df['endNode'][i]] = df['forwardCap'][i]
    backwardCapDict[df['endNode'][i]][df['startNode'][i]] = df['backwardCap'][i]
    distanceDict[df['startNode'][i]][df['endNode'][i]] = df['Length'][i]
    forwardSpeedDict[df['startNode'][i]][df['endNode'][i]] = df['forwardSpeed'][i]
    backwardSpeedDict[df['endNode'][i]][df['startNode'][i]] = df['backwardSpeed'][i]


counter = 1
connections = []
reverseConnections = []
for i in trafficInfoDict['edgeID']:
    forwardList = []
    reverseList = []
    a = int(trafficInfoDict['startNode'][i])
    b = int(trafficInfoDict['endNode'][i])
    forwardList.append(a)
    forwardList.append(b)
    reverseList.append(b)
    reverseList.append(a)
    d = tuple(forwardList)
    e = tuple(reverseList)
    connections.append(d)
    reverseConnections.append(e)

proximityDict = Graph(connections, directed=True)

capacityMatrix = {}
capacityMatrix = defaultdict(dict)
distanceMatrix = {}
distanceMatrix = defaultdict(dict)
speed = {}
speedMatrix = defaultdict(dict)

for i in range(1, len(proximityDict._graph)+1):
    capacityMatrix[i][i] = 0
    distanceMatrix[i][i] = 0
    for j in range(1, len(proximityDict._graph)+1):
        temp = []
        if i < j:
            temp = [i,j]
            if tuple(temp) in connections:
                capacityMatrix[i][j] = forwardCapDict[i][j]
                distanceMatrix[i][j] = distanceDict[i][j]
                distanceMatrix[j][i] = distanceDict[i][j]
                speedMatrix[i][j] = forwardSpeedDict[i][j]
                
    
            else:
                capacityMatrix[i][j] = 0
                distanceMatrix[j][i] = 0
                distanceMatrix[i][j] = 0
                speedMatrix[i][j] = 0
        else:
            temp = [i,j]
            if tuple(temp) in reverseConnections:
                capacityMatrix[i][j] = backwardCapDict[i][j]
                speedMatrix[i][j] = backwardSpeedDict[i][j]
            else:
                capacityMatrix[i][j] = 0
                speedMatrix[i][j] = 0

capacitydf = pd.DataFrame(capacityMatrix).T
distancedf = pd.DataFrame(distanceMatrix).T
speeddf = pd.DataFrame(speedMatrix).T
distancedf.fillna(0)
flowdf = pd.DataFrame(0, index=np.arange(1, len(capacitydf)+1), columns=np.arange(1, len(capacitydf)+1))
adjMatrixdf = pd.DataFrame(0, index=np.arange(1, len(capacitydf)+1), columns=np.arange(1, len(capacitydf)+1))
adjMatrixdf = adjMatrix(capacitydf, flowdf) 

allNodes = []
for m in df['startNode']:
    if m not in allNodes:
        allNodes.append(m)
        
for n in df['endNode']:
    if n not in allNodes:
        allNodes.append(n)
        
# Give/ASK FOR sources and targets
print "Select Sources from 1 to %d." %(len(allNodes))
sources= list(input("Input Sources (Write as a list for a single Source) :" ))
for i in sources:
    if i not in allNodes:
        print "ERROR: Source %d is not in nodeList, give the values between 1 to %d" % (i, len(allNodes))
        exit()

print "Select Targets from 1 to %d." %(len(allNodes))        
targets= list(input("Input Targets (Write as a list for a single Target) :")) 
for i in targets:
    if i not in allNodes:
        print "ERROR: Target %d is not in nodeList, give the values between 1 to %s" % (i, len(allNodes))
        exit()
# sources = [1,8,16]
# targets = [45,58,62]

newCap = capacitydf
newDis = distancedf
newAdj = adjMatrixdf

# Add virtual source and node if needed
if (len(targets)==1 and len(sources)>1 ):
    s = len(newCap.index)+1
    newCap[s]=0.0
    newAdj[s]=0.0
    newCap.loc[s] = 0.0
    newAdj.loc[s]=0.0
    newDis[s]=0.0
    newDis.loc[s]=0.0
    for i in sources:
        newCap[i][s]= 100000000
        newAdj[i][s]=1
elif (len(sources)==1 and len(targets)>1 ):
    t = len(newCap.columns)+1
    newCap[t]=0.0
    newAdj[t]=0.0
    newCap.loc[t] = 0.0
    newAdj.loc[t]=0.0
    newDis[t]=0.0
    newDis.loc[t]=0.0
    for i in targets:
        newCap[t][i]= 100000000
        newAdj[t][i]=1
elif (len(sources)>1 and len(targets)>1 ):
    s = len(newCap.index)+1
    t = s+1
    newCap[s]=0.0
    newAdj[s]=0.0
    newCap.loc[s] = 0.0
    newAdj.loc[s]=0.0
    newDis[s]=0.0
    newDis.loc[s]=0.0
    newCap[t]=0.0
    newAdj[t]=0.0
    newCap.loc[t] = 0.0
    newAdj.loc[t]=0.0
    newDis[t]=0.0
    newDis.loc[t]=0.0
    for i in sources:
        newCap[i][s]= 100000000
        newAdj[i][s]=1
    for i in targets:
        newCap[t][i]= 100000000
        newAdj[t][i]=1

newFlow = pd.DataFrame(0, index=np.arange(1, len(newCap)+1), columns=np.arange(1, len(newCap)+1))

if len(sources) == 1:
    s = sources[0]
if len(targets) == 1:
    t = targets[0]

# See how many nodes are there in our network
allNodes = []
for m in df['startNode']:
    if m not in allNodes:
        allNodes.append(m)
        
for n in df['endNode']:
    if n not in allNodes:
        allNodes.append(n)


# Call the main ford-fulkersion function to get required details
[newAdj, p, lab, maxFlow, newFlow, minCuts, minCutDict, minCut, pathDict] = fordFulkersion(newCap, newFlow, adjMatrixdf, s, t)

#print "PATHDICT",pathDict
print "\nMAXFLOW: ", maxFlow
print "\nMINCUT: ", minCut
print "\nMINCUT EDGES: ", minCuts

# To somve the last redundunt path
del pathDict[len(pathDict)]
if (len(targets)==1 and len(sources)>1 ):
    for i in range(1,len(pathDict)+1):
        if t in pathDict[i]:
            pathDict[i].remove(t)
            
elif (len(sources)==1 and len(targets)>1 ):           
    for i in range(1,len(pathDict)+1):        
        if s in pathDict[i]:
            pathDict[i].remove(s)
            
elif (len(sources)>1 and len(targets)>1 ):
    for i in range(1,len(pathDict)+1):            
        if t in pathDict[i]:
            pathDict[i].remove(t)
        if s in pathDict[i]:
            pathDict[i].remove(s)
        
edgeList= []
for i in range(1,len(pathDict)+1):
    for j in range(0,len(pathDict[i])-1):
        #print pathDict[i]
        a = pathDict[i][j]
        b = pathDict[i][j+1]
        k = [a, b]
        l = tuple(k)
        if l not in edgeList:
            edgeList.append(l)

# Get longitude and latitude of the data points


# Make a list of the unique edges, their capacity, flow and slacks of s-t quasi path
edges = []
edgeFlow = []
edgeCapacity = []
slacks = []
for i in range(0, len(edgeList)):
    edgeFlow.append(newFlow.loc[edgeList[i][0]][edgeList[i][1]])
    edgeCapacity.append(newCap.loc[edgeList[i][0]][edgeList[i][1]])
    edges.append(edgeList[i])
# To make sure that each point should be oplotted only once


# Make a dataframe showing each edgeflow, capacity and slacks
edgeFlowdf = pd.DataFrame()
edgeFlowdf['edges'] = edges
edgeFlowdf['Flow'] = edgeFlow
edgeFlowdf['Capacity'] = edgeCapacity
edgeFlowdf['slack'] = edgeFlowdf['Capacity'] - edgeFlowdf['Flow']

edgeFlowdf.index = edgeFlowdf.index + 1 
print "\nFLOW THROUGH EDGES"
print edgeFlowdf
print "Path Id \t Path"
for i in pathDict:
    print "%d     \t %s" %(i, pathDict[i])
# save the required matrix 
newCap.to_csv('capacityData')
newFlow.to_csv('finalFlowData')
newAdj.to_csv('adjData')
edgeFlowdf.to_csv('edgesData')

# plot the turn by turn directions, flow on each edges and nodes itself
markerCheck = []


if sys.argv[1] != 'trafficData':
    fileCSV3 = 'BuffaloNew.xlsx'
    buffaloNew= pd.read_excel(fileCSV3, sheet = 1, encoding="utf-8")
    buffaloNew.index = buffaloNew.index + 1 
     
    # Create a map to show how the flow looks like  on real road network 
    mapFile = 'Buffaloflow.html'
    map_osm = folium.Map(location=[buffaloNew.loc[1]['Lat'], buffaloNew.loc[1]['Long']], zoom_start=12)
    for i in range(0, len(edgeList)):
        startCoords = '%f,%f' % (buffaloNew.loc[edgeList[i][0]]['Lat'], buffaloNew.loc[edgeList[i][0]]['Long'])
        endCoords = '%f,%f' % (buffaloNew.loc[edgeList[i][1]]['Lat'], buffaloNew.loc[edgeList[i][1]]['Long'])      
        myShapepoints = genShapepoints(startCoords, endCoords)
        if newFlow.loc[edgeList[i][0]][edgeList[i][1]] <= 500:           
            folium.PolyLine(myShapepoints, color="Green", weight=6, opacity=0.5).add_to(map_osm)
        elif newFlow.loc[edgeList[i][0]][edgeList[i][1]] > 500 and newFlow.loc[edgeList[i][0]][edgeList[i][1]] <= 1500:   
            folium.PolyLine(myShapepoints, color="Yellow", weight=6, opacity=0.5).add_to(map_osm)
        elif newFlow.loc[edgeList[i][0]][edgeList[i][1]] > 1500 and newFlow.loc[edgeList[i][0]][edgeList[i][1]] <= 2500:   
            folium.PolyLine(myShapepoints, color="Blue", weight=6, opacity=0.5).add_to(map_osm)
        else:
            folium.PolyLine(myShapepoints, color="Red", weight=6, opacity=0.5).add_to(map_osm)
             
             
        for m in edgeList[i]:
            if m not in markerCheck:
                #print m, buffaloNew.loc[m]['Lat'], buffaloNew.loc[m]['Long']
                folium.Marker([buffaloNew.loc[m]['Lat'], buffaloNew.loc[m]['Long']],  icon=DivIcon('<div style="font-size: 15pt">'+str(m)+'</div>',) ).add_to(map_osm)
                if m in sources:
                    folium.Marker([buffaloNew.loc[m]['Lat'], buffaloNew.loc[m]['Long']],  icon=folium.Icon(color='orange'), popup = 'Source' ).add_to(map_osm)
                elif m in targets:
                    folium.Marker([buffaloNew.loc[m]['Lat'], buffaloNew.loc[m]['Long']],  icon=folium.Icon(color='purple'), popup = 'Target').add_to(map_osm)
                else:
                    folium.Marker([buffaloNew.loc[m]['Lat'], buffaloNew.loc[m]['Long']],  icon=folium.Icon(color='blue') ).add_to(map_osm)
                markerCheck.append(m)
    
    # Highlight the mincut edges
    for i in range(0, len(minCuts)):
        startCoords = '%f,%f' % (buffaloNew.loc[minCuts[i][0]]['Lat'], buffaloNew.loc[minCuts[i][0]]['Long'])
        endCoords = '%f,%f' % (buffaloNew.loc[minCuts[i][1]]['Lat'], buffaloNew.loc[minCuts[i][1]]['Long'])      
        myShapepoints = genShapepoints(startCoords, endCoords)        
        folium.PolyLine(myShapepoints, color="Red", weight=10, opacity=0.8).add_to(map_osm)
        
# Save the map        
    map_osm.save(mapFile)     
    #Open the map in web browser
    new = 1;
    url = 'C:\Users\Test\Desktop\desk docs\Study\MSIS\SEM 2\PFA\Eclipse\PFA\Buffaloflow.html';
    webbrowser.open(url, new=new)   


 


