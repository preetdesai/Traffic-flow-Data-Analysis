from folium.features import DivIcon
import sys            # Allows us to capture command line arguments
import csv
import folium        # https://github.com/python-visualization/folium

import urllib2
import json
import pandas as pd
from pandas.io.json import json_normalize
import webbrowser 
from collections import defaultdict

mapquestKey = 'tKFF3XDTAOzJ8GYjUtVkIQKkCrbuUsXj'

def genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr):
    # We'll use MapQuest to calculate.
    transportMode = 'fastest'
    # Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)
    routeTypeStr = 'routeType:%s' % transportMode

    # Assemble query URL
    myUrl = 'http://www.mapquestapi.com/directions/v2/routematrix?'
    myUrl += 'key={}'.format(mapquestKey)
    myUrl += '&inFormat=json&json={locations:['

    # Insert coordinates into the query:
    n = len(coordList)
    for i in range(0,n):
        if i != n-1:
            myUrl += '{{latLng:{{lat:{},lng:{}}}}},'.format(coordList[i][0], coordList[i][1])
        elif i == n-1:
            myUrl += '{{latLng:{{lat:{},lng:{}}}}}'.format(coordList[i][0], coordList[i][1])
    myUrl += '],options:{{{},{},{},{},doReverseGeocode:false}}}}'.format(routeTypeStr, all2allStr,one2manyStr,many2oneStr)

    # print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
    # print myUrl


    # Now, we'll let Python go to mapquest and request the distance matrix data:
    request = urllib2.Request(myUrl)
    response = urllib2.urlopen(request)
    data = json.loads(response.read())

    # print "\nHere's what MapQuest is giving us:"
    # print data

    # This info is hard to read.  Let's store it in a pandas dataframe.
    # We're goint to create one dataframe containing distance information:
    distance_df = json_normalize(data, "distance")
    # print "\nHere's our 'distance' dataframe:"
    # print distance_df

    # print "\nHere's the distance between the first and second locations:"
    # print distance_df.iat[0,1]

    # Our dataframe is a nice table, but we'd like the row names (indexes)and column names to match our location IDs.
    # This would be important if our locationIDs are [1, 2, 3, ...] instead of [0, 1, 2, 3, ...]
    distance_df.index = locIDlist
    distance_df.columns = locIDlist

    # Now, we can find the distance between location IDs 1 and 2 as:
    # print "\nHere's the distance between locationID 1 and locationID 2:"
    # print distance_df.loc[1,2]


    # We can create another dataframe containing the "time" information:
    time_df = json_normalize(data, "time")

    # print "\nHere's our 'time' dataframe:"
    # print time_df

    # Use our locationIDs as row/column names:
    time_df.index = locIDlist
    time_df.columns = locIDlist


    # We could also create a dataframe for the "locations" information (although we don't need this for our problem):
    #print "\nFinally, here's a dataframe for 'locations':"
    #df3 = json_normalize(data, "locations")
    #print df3

    return(distance_df, time_df)


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

fileCSV = 'C:\Users\Test\Desktop\desk docs\Study\MSIS\SEM 2\PFA\Eclipse\PFA\BuffaloNew.xlsx'

df = pd.read_excel(fileCSV, sheet = 2, encoding="utf-8")
df.index = df.index + 1

print df



mapFile = 'C:\Users\Test\Desktop\desk docs\Study\MSIS\SEM 2\PFA\Eclipse\PFA\BuffaloPoints.html'
map_osm = folium.Map(location=[df.iloc[1][1], df.loc[1][2]], zoom_start=10)

nodes = []

for i in range(1, len(df)+1):
    folium.Marker([df.loc[i][1], df.loc[i][2]],  icon=DivIcon('<div style="font-size: 15pt">'+str(i)+'</div>',) ).add_to(map_osm)
    folium.Marker([df.loc[i][1], df.loc[i][2]],  icon=folium.Icon(color='blue') ).add_to(map_osm)
    nodes.append(i)
    
nodes.append(1)

#print nodes
map_osm.save(mapFile)
# 
#Open the map in web browser
new = 1;
url = 'C:\Users\Test\Desktop\desk docs\Study\MSIS\SEM 2\PFA\Eclipse\PFA\BuffaloPoints.html';
webbrowser.open(url, new=new)

exit()
fileCSV2 = 'C:\Users\Test\Desktop\desk docs\Study\MSIS\SEM 2\PFA\Eclipse\PFA\BuffaloAdj.xlsx'

dfbuffalo= pd.read_excel(fileCSV2, sheet = 1, encoding="utf-8")

dfbuffalo.index = dfbuffalo.index + 1

print dfbuffalo


#Generate distance matrix (a pandas dataframe)
forwardDict = []
reverseDict = []
forwardTime = []
reverseTime = []
for i in range(1, len(dfbuffalo)+1):
    coordList = []
    locIDlist = []
    coordList.append([df.loc[dfbuffalo.loc[i]['startNode']]['Lat'], df.loc[dfbuffalo.loc[i]['startNode']]['Long']])
    coordList.append([df.loc[dfbuffalo.loc[i]['endNode']]['Lat'], df.loc[dfbuffalo.loc[i]['endNode']]['Long']])
    locIDlist.append(i)
    locIDlist.append(i+1)
    all2allStr    = 'allToAll:true'
    one2manyStr    = 'oneToMany:false'
    many2oneStr    = 'manyToOne:false'
    [df1, df2] = genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr)
    forwardDict.append(df1.loc[i][i+1])
    reverseDict.append(df1.loc[i+1][i])
    forwardTime.append(df2.loc[i][i+1])
    reverseTime.append(df2.loc[i+1][i])
    dfDist = df1.to_dict(orient='index')
    dfTime = df2.to_dict(orient='index')
   
    
dfbuffalo['forwardDist'] = forwardDict
dfbuffalo['reverseDist'] = reverseDict

dfbuffalo['forwardTime'] = forwardTime
dfbuffalo['reverseTime'] = reverseTime

dfbuffalo.to_csv('dfBuffalo')

# i = nodes[0]
# for j in nodes[1:]:
#     startCoords = '%f,%f' % (df.loc[i][1], df.loc[i][2])
#     endCoords = '%f,%f' % (df.loc[j][1], df.loc[j][2])         
#     myShapepoints = genShapepoints(startCoords, endCoords)           
#     folium.PolyLine(myShapepoints, color="blue", weight=6, opacity=0.5).add_to(map_osm)    
#     i = j
     
print dfbuffalo
