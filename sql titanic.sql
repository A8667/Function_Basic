select* from titanic;
 select age as gender ,fare,pclass from titanic;
 select *from name;
 
 select count(*) from titanic;	
  select count(age) from titanic;	
  select distinct age from titanic;
  select count(distinct age) from titanic;
    select count(age) from titanic where fare =151.55;	
    
    select avg(age) from titanic;
	select max(age) from titanic;	
    
    select age,count(*)
    from titanic
    group by age;