input_data=(62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# #change input data to a numpy array
# input_data_np_array=np.asarray(input_data)

# # reshape the numpy array as we are predicting for only one instance
# input_data_reshape=input_data_np_array.reshape(1,-1)

# prediction = model.predict(input_data_reshape)
# print(prediction)

# if(prediction[0]==0):
#   print("The person does not have heart disease")
# else:
#   print("The person has heart disease")