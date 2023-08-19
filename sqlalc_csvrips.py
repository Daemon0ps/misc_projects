if __name__ == '__main__':
    fp="./"
    import os
    from keyring import get_password as GP
    import pandas as _p
    from sqlalchemy import create_engine as CE,URL,text as _t
    from sqlalchemy.orm import sessionmaker as SM,close_all_sessions as CA
    if not os.path.isdir(f"{fp}/schema"):
        os.makedirs(f"{fp}/schema")
    if not os.path.isdir(f"{fp}/csv"):
        os.makedirs(f"{fp}/csv")
    cx_u=lambda x:URL.create(
    drivername="mssql+pyodbc",username=GP("0x0c_sql","user"),
    password=GP("0x0c_sql","pass"),host=GP("0x0c_sql","server"),
    database=x,port=GP("0x0c_sql","port_num"),
    query={"driver":"ODBC Driver 18 for SQL Server","TrustServerCertificate":"yes",})
    cn=lambda x:CE(url=cx_u(x))
    cr=lambda x:SM(bind=CE(url=cx_u(x)))().connection().connection.cursor()
    qry=str(r"select name from sys.databases where name in('master','tempdb','model','msdb')")
    n=_p.read_sql(sql=_t(qry),con=cn('master'))
    l=n.copy().astype('str').to_numpy().reshape(-1).tolist()
    for x in l:
        if not os.path.isdir(f"{fp}/schema/{x}"):
            os.makedirs(f"{fp}/schema/{x}")
        if not os.path.isdir(f"{fp}/csv/{x}"):
            os.makedirs(f"{fp}/csv/{x}")
        qry=str(r"select * from information_schema.columns")
        i_s=_p.read_sql(sql=_t(qry),con=cn(x))
        i_s.to_csv(f"{fp}/schema/{x}/{x}_schema.csv")
        qry=str(r"select table_name from information_schema.tables")
        t=_p.read_sql(sql=_t(qry),con=cn(x))
        b=t.copy().astype('str').to_numpy().reshape(-1).tolist()
        for d in b:
            qry=str(f"select * from [{x}].dbo.[{d}]")
            with cn(x).connect() as cu:
                res=cu.execute(_t(qry))
                _p.DataFrame(res).to_csv(f"{fp}/csv/{x}/{d}.csv")
                cu.close()
        CA()
