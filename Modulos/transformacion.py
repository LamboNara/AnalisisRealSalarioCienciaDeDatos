def transform():
    df= archivo.drop(columns= "Unnamed: 0")

    #Cambiar nombres de categorías para mayor comprensión
    # Remote ratio
    df.remote_ratio.replace([100,50,0],['Fully Remote','Partially Remote','No remote work'], inplace=True)

    # Experience Level
    df.experience_level.replace(['EN','MI','SE','EX'],['Entry','Junior','Senior','Expert'],inplace=True)

    # Employment Type
    df.employment_type.replace(['FT','PT','CT','FL'],['Full Time','Part Time','Contract','Freelance'], inplace=True)

    #Company size

    df.company_size.replace(['S','M','L'],['Small','Medium','Large'], inplace=True)

transform()