''' 
Indexes result_struct.json in the elastic search server
'''
import elasticsearch
import json
#Instantiates an elastic search type object which has various utility functions such as index,search
es = elasticsearch.Elasticsearch()  
#opens the json file and converts it to python dictionary 
with open("result_struct.json") as f:
    data = json.load(f)
#counts the number of images present in the json file 
num_rec = len(data['imgblobs']) ## no of imgs in result_struct.json file 

#the python dictionary which we are going to indexi n the elasticsearch server
new_d = [ {} for _ in xrange(num_rec)]
#iterates through every image,extracts only the imgurl and the description
for i in xrange(num_rec):
    new_d[i]['imgurl'] = data['imgblobs'][i]['img_path']
    new_d[i]['description'] = data['imgblobs'][i]['candidate']['text']
#now go through every image, indexes them in the server
for i in xrange(num_rec):
    es.index(index="desearch", doc_type="json", id=i, body = {
                    'imgurl': new_d[i]['imgurl'],
                    'description': new_d[i]['description'],
                    'idnum': i
                })
