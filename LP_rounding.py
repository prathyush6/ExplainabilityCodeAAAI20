from gurobipy import *
import argparse
from datetime import datetime
import random as rand
# import input_preprocessing
# import ecml_data_processing
import pandas as pd
import numpy as np

def create_model_ilp(input_params,map_args):
  print("-------------------------------------------------------------------------------------------------------------")
  print("Starting ILP")
  print("-------------------------------------------------------------------------------------------------------------")
  ilp_starttime=datetime.now()
  m=Model('threat_ilp')

  # Create variables. - variable names is same as attribute names- for coding purposes and not for mathematical.
  X=[]
  Y=[]
  Z=[]

  for i in range(0, len(input_params['attribute_set'])):
      X.append(m.addVar(vtype=GRB.BINARY, name="X"+str(i)))
  for i in range(0, len(input_params['attribute_set'])):
      Y.append(m.addVar(vtype=GRB.BINARY, name="Y"+str(i)))

  for i in range(0,len(input_params['dataset'])):
    Z.append(m.addVar(vtype=GRB.BINARY, name="Z"+str(i)))

  # Constraints.
  # Add budget constraint - No of total attributes selected should be <=budget.
  m.addConstr(quicksum(X[i]+Y[i] for i in range(0, len(input_params['attribute_set'])))<=int(map_args.budget))
  for i in input_params['cluster_1_index']:
    m.addConstr(quicksum(input_params['dataset_matrix'][i][j] * X[j] for j in range(0, len(input_params['attribute_set']))) >= Z[i])


  for i in input_params['cluster_2_index']:
    m.addConstr(quicksum(input_params['dataset_matrix'][i][j] * Y[j] for j in range(0, len(input_params['attribute_set']))) >= Z[i])

  # condition which satisfies that the two sets should be disjoint sets.
  for j in range(0, len(input_params['attribute_set'])):
      m.addConstr(X[j] + Y[j] <= 1)

  m.addConstr(quicksum(Z[i] for i in input_params['cluster_1_index']) >= int(map_args.M1))
  m.addConstr(quicksum(Z[i] for i in input_params['cluster_2_index']) >= int(map_args.M2))

  m.setObjective(quicksum(Z[i] for i in range(0,len(input_params['dataset']))), GRB.MAXIMIZE)
  #m.setObjective(quicksum(X[i]+Y[i] for i in range(0, len(input_params['attribute_set']))), GRB.MINIMIZE)
  m.update()
  m.write('ilp_icml.lp')
  # m.setParam(GRB.Param.MIPGap,0.1)
  m.optimize()
  if m.status==GRB.Status.OPTIMAL:
    ilp_endtime=datetime.now()
    count=0
    for i in range(0, len(input_params['attribute_set'])):
      if X[i].X>0:
        count+=1
        print(X[i].VarName)

    c1_count=count
    print("Attributes selected from Cluster 1", c1_count)
    for i in range(0, len(input_params['attribute_set'])):
      if Y[i].X>0:
        count+=1
        print(Y[i].VarName)

    c1_hit_count=0
    c2_hit_count=0
    for i in range(0, len(input_params['dataset'])):
      if Z[i].X>0:
        if i in input_params['cluster_1_index']:
          c1_hit_count+=1
        elif i in input_params['cluster_2_index']:
          c2_hit_count+=1

    print("Attributes selected from Cluster 2", count-c1_count)
    print("# of attributes picked = ", count)
    print("# of elements hit in cluster 1: ", c1_hit_count)
    print("# of elements hit in cluster 2: ", c2_hit_count)
  else:
    print("No solution.")

  print("-------------------------------------------------------------------------------------------------------------")
  print("ILP Finished.")
  print("-------------------------------------------------------------------------------------------------------------")


def create_model_lp(input_params,map_args):
  # Gurobi model implementation.
  # Create a model.
  print("-------------------------------------------------------------------------------------------------------------")
  print("Starting LP")
  print("-------------------------------------------------------------------------------------------------------------")
  print("Gurobi Model Creation!")
  start_time=datetime.now()

  m=Model('lp_icml')
  # Create decision variables. - variable names is same as attribute names- for coding purposes and not for mathematical.
  X=[]
  Y=[]
  Z=[]

  for i in range(0, len(input_params['attribute_set'])):
    X.append(m.addVar(lb=0.0, ub=1.0, name="X"+str(i)))
  for i in range(0, len(input_params['attribute_set'])):
    Y.append(m.addVar(lb=0.0, ub=1.0, name="Y"+str(i)))

  for i in range(0, len(input_params['dataset'])):
    Z.append(m.addVar(lb=0.0, ub=1.0, name="Z"+str(i)))

  m.addConstr(quicksum(X[i]+Y[i] for i in range(0, len(input_params['attribute_set'])))<=int(map_args.budget))  # int(budget)

  for i in input_params['cluster_1_index']:
    m.addConstr(quicksum(input_params['dataset_matrix'][i][j]*X[j] for j in range(0, len(input_params['attribute_set'])))>=Z[i])

  for i in input_params['cluster_2_index']:
    m.addConstr(quicksum(input_params['dataset_matrix'][i][j]*Y[j] for j in range(0, len(input_params['attribute_set'])))>=Z[i])

  # condition which satisfies that the two sets should be disjoint.
  for j in range(0, len(input_params['attribute_set'])):
    m.addConstr(X[j]+Y[j]<=1)

  m.addConstr(quicksum(Z[i] for i in input_params['cluster_1_index'])>=int(map_args.M1))
  m.addConstr(quicksum(Z[i] for i in input_params['cluster_2_index'])>=int(map_args.M2))

  m.setObjective(quicksum(Z[i] for i in range(0, len(input_params['dataset']))), GRB.MAXIMIZE)
  #m.setObjective(quicksum(X[i]+Y[i] for i in range(0, len(input_params['attribute_set']))), GRB.MINIMIZE)

  m.update()
  m.write('lp_icml.lp')
  # m.setParam(GRB.Param.MIPGap, 0.2)
  m.optimize()
  if m.status == GRB.Status.OPTIMAL:
    lp_endtime = datetime.now()
    print('\nLP objective value: %g'%m.objVal,'LP runtime: {}'.format(lp_endtime - start_time))
    r_starttime=datetime.now()
    X_star = []
    Y_star = []
    Z_star = []
    for j in range(0, len(input_params['attribute_set'])):
      X_star.append(X[j].X)
      Y_star.append(Y[j].X)
    for i in range(0, len(input_params['dataset'])):
      Z_star.append(Z[i].X)

    #print("X_star", X_star)
    #print("Y_star", Y_star)
    iteration_results = {}
    X1=[]
    X1_attr=[]  # selected attribute names for cluster 1.
    Y1=[]
    Y1_attr=[]  # selected attribute names for cluster 2.
    for k in range(0, int(map_args.iterations)):
      X1=[]
      X1_attr=[]  # selected attribute names for cluster 1.
      Y1=[]
      Y1_attr=[]  # selected attribute names for cluster 2.
      Z1=[]  # for cluster 1.
      Z2=[]  # for cluster 2.
      no_of_hits_z2=0
      no_of_hits_z1=0
      budget_check = 0
      for i in range(0, len(input_params['attribute_set'])):
        X1.append(0)
        Y1.append(0)
      for j in range(0, len(input_params['attribute_set'])):
        vals = []
        #print("Tag", j)
        vals.append(X_star[j])
        vals.append(Y_star[j])
        vals_cum = np.cumsum(vals)
        #print("Cumulative: ", vals_cum)
        random_num_1 = rand.random()
        #print(random_num_1)
        flag = 0
        cluster = 0
        for i in range(len(vals_cum)):
          #print("i = ", i)
          if random_num_1 < vals_cum[i]:
            #print("Condition satisfied")
            flag = 1
            cluster = i + 1
            break

        if flag == 1:
          if cluster == 1:
            #print("Tag selected in cluster 1")
            X1[j]=1
            Y1[j]=0
            budget_check+=1
            X1_attr.append(X[j].VarName)

          elif cluster == 2:
            #print("Tag selected in cluster 2")
            Y1[j]=1
            X1[j]=0
            budget_check+=1
            Y1_attr.append(Y[j].VarName)

          #   #print("Inside if")
          #   print("Tag selected in cluster 1")
          #   X1[j]=1
          #   Y1[j]=0
          #   budget_check+=1
          #   X1_attr.append(X[j].VarName)
          #   #break
          # elif random_num_1 <= vals_cum[i]:
          #   #print("Inside else if")
          #   print("Tag selected in cluster 2")
          #   Y1[j]=1
          #   X1[j]=0
          #   budget_check+=1
          #   Y1_attr.append(Y[j].VarName)
          #   #break
          # else:
          #   #print("Inside else")
          #   print("Tag rejected")
          #   X1[j]=0
          #   Y1[j]=0

       #print("i here "+str(i))
      #print("X1", X1)
      #print("Y1", Y1)
      for i in input_params['cluster_1_index']:
        elem = input_params['dataset'].iloc[[i]]
        flag = False
        # probability = 1 if there exist a j such that X1[j] = 1.
        for column in elem:
          #print("Column index:", elem.columns.get_loc(column))
          index = elem.columns.get_loc(column)
          if X1[index] == 1 and elem[column].values[0] == 1:
            flag = True
            break
        if flag:
          Z1.append(1)
          no_of_hits_z1+=1
        else:
          Z1.append(0)

      # Cluster 2.
      for i in input_params['cluster_2_index']:
        elem = input_params['dataset'].iloc[[i]]
        flag = False
        # probability = 1 if there exist a j such that X1[j] = 1.
        for column in elem:
          #print("Column index:", elem.columns.get_loc(column))
          index = elem.columns.get_loc(column)
          if Y1[index] == 1 and elem[column].values[0] == 1:
            flag = True
            break
        if flag:
          Z2.append(1)
          no_of_hits_z2+=1
        else:
          Z2.append(0)

      obj_value = no_of_hits_z1 + no_of_hits_z2
      if budget_check <= (2 * int(map_args.budget)) and no_of_hits_z1 >= (int(map_args.M1) / 8.0) and no_of_hits_z2 >= (int(map_args.M2) / 8.0):
        iteration_results[k] = {"Z1" : Z1, "Z2" : Z2, "obj_value" : obj_value,'X1' : X1,'Y1' : Y1, 'X1_attr': X1_attr, 'Y1_attr':Y1_attr,'attr_selected':(len(X1_attr)+len(Y1_attr))}
    lp_endtime = datetime.now()
    print('LP Rounding 1 runtime: {}'.format(lp_endtime-start_time))
    print("Processing LP Rounding 1 Results.... ")
    obj_value = 0
    curr_high_value = 0
    high_key = -1
    print("# of candidate solutions: ", len(iteration_results.keys()))
    for key in iteration_results:
      #obj_value = iteration_results[key]['attr_selected']
      obj_value = iteration_results[key]['obj_value']
      print("Attributes selected: ", key, obj_value)
      if obj_value > curr_high_value :
        curr_high_value = obj_value
        high_key = key
    print(high_key)
    print("LP Rounding 1 Objective value: ", iteration_results[high_key]['obj_value'])
    count_final_z1 = 0
    for i in iteration_results[high_key]['Z1']:
      if i == 1:
        count_final_z1 += 1
    print("Total # of hits in cluster 1 after Rounding 1: ", count_final_z1)
    count_final_z2 = 0
    for i in iteration_results[high_key]['Z2']:
      if i == 1:
        count_final_z2 += 1
    print("Total # of hits in cluster 2 after Rounding 1: ", count_final_z2)
    c1_attr = 0
    #X1_attr_selected = []
    for i in iteration_results[high_key]['X1']:
      if i == 1:
        c1_attr += 1
        #X1_attr_selected.append((iteration_results[high_key]['X1_attr']).index(iteration_results[high_key]['X1_attr'][i]))
    print("# of attributes selected in C1: ", c1_attr)
    c2_attr = 0
    #X2_attr_selected=[]
    for i in iteration_results[high_key]['Y1']:
      if i == 1:
        c2_attr += 1
        #X2_attr_selected.append((iteration_results[high_key]['Y1_attr']).index(iteration_results[high_key]['Y1_attr'][i]))
    print("# of attributes selected in C2: ", c2_attr)
    print("Attribute selected in C1 after rounding 1: ", iteration_results[high_key]['X1_attr'])
    print("Attribute selected in C2 after rounding 1: ", iteration_results[high_key]['Y1_attr'])
    r_endtime=datetime.now()
    print('LP runtime: {}'.format(r_starttime - r_endtime))
  else:
    print("No solution.")

  print("-------------------------------------------------------------------------------------------------------------")
  print("LP Done.")
  print("-------------------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
  # Read command line arguments.
  parser=argparse.ArgumentParser()
  parser.add_argument('budget', type=str, help="input Budget")
  parser.add_argument('M1', type=str, help="input M1 value")
  parser.add_argument('M2', type=str, help="input M2 value")
  #parser.add_argument('input_file_pair_of_tags', type=str, help="Pair-of-tags input file path")
  parser.add_argument('input_file', type=str, help="Input file path")
  parser.add_argument('iterations', type=str, help="# of iterations for LP Rounding.")
  #parser.add_argument('dataset', type=str, help="synthetic pulse-check philosopher")
  if len(sys.argv) == 0:
    sys.exit(2)
  map_args=parser.parse_args()

  # Step 1: Pre-process Input.
  #input_preprocessing.process_input(map_args)
  #ecml_data_processing.process_synthetic_dataset(map_args)
  #input_params = ecml_data_processing.process_philosopher_dataset(map_args)
  #input_params = input_preprocessing.process_synthetic_dataset(map_args.input_file)
  #input_params=ecml_data_processing.process_synthetic_dataset(map_args)
  #print(input_params)
  # Step 2: Run ILP
  #create_model_ilp(input_params,map_args)
  # Step 3: Run LP with Rounding 1

  dataset=pd.read_csv(map_args.input_file)

  dataset= dataset.drop(['E',], axis=1)
  #dataset=dataset.drop(
     #['Seq_Id', 'swiss-prot', 'GO:0044419', 'KW-0181', 'GO:0051704', 'KW-1185', 'GO:0009405', 'GO:0005488', 'GO:0005576',
     # 'GO:0009987', 'GO:0090729', 'KW-0800', 'GO:0008152', 'GO:0003824', 'KW-0964'], axis=1)
  #dataset = dataset.drop(['E'],axis = 1)
  cluster_1_index=[]
  cluster_2_index=[]
  for index, row in dataset.iterrows():
    #if row['threat_bin']==1 or row['threat_bin']==2:
    #if row['cluster'] == 1:
    if row['C'] == 0:
    #if row['C'] == 'C1':
      cluster_1_index.append(index)
    else:
      cluster_2_index.append(index)
  #dataset=dataset.drop(['threat_bin'], axis=1)
  dataset = dataset.drop(['C'], axis=1)
  #dataset = dataset.drop(['cluster'],axis = 1)
  print("# of elements in cluster 1:", len(cluster_1_index))
  print("# of elements in cluster 2:", len(cluster_2_index))

  #dataset_new = pd.read_csv(map_args.input_file_pair_of_tags)
  #attribute_set=dataset_new.columns.values.tolist()
  attribute_set = dataset.columns.values.tolist()
  print("# of attributes: ", len(attribute_set))
  #dataset_matrix=dataset_new.as_matrix()
  dataset_matrix = dataset.as_matrix()

  input_params={'cluster_1_index': cluster_1_index, 'cluster_2_index': cluster_2_index,
                'attribute_set': attribute_set, 'dataset': dataset,
                'dataset_matrix': dataset_matrix}

  create_model_ilp(input_params, map_args)
  #create_model_lp(input_params, map_args)

