from prefect import flow, task
@task
def addition():
    print('Addition function done')
@flow
def main():
    addition()

main()