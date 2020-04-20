Animal Farm Project

> Don't let the farm run by pigs

This project is built for practice purpose only.

## Setup dev env

### Create mysql container, set privileges (keep in mind granding with % is dangerous in production)

```
docker run -it -d --name af_mysql --env MYSQL_ROOT_PASSWORD=xxx mysql:8

docker exec -it af_mysql bash
CREATE USER 'af'@'%' IDENTIFIED BY 'xxx';
GRANT ALL PRIVILEGES ON *.* TO 'af'@'%';
```

### Create app container

#### Build base docker image for app
```
docker build -t af_base_img -f Dockerfile_app_base_img .
```

#### Build app container using base image
```
docker run -it -d --name af_app -p 8080:80 -v /home/mguo/git/animal_farm:/code --link af_mysql af_base_img
```