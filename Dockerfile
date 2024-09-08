FROM public.ecr.aws/lambda/python:3.9

COPY . ${LAMBDA_TASK_ROOT}

RUN yum install libgomp -y
RUN pip install -r requirements.txt

CMD [ "app.handler" ]