from multiprocessing import Process, Semaphore


def worker(semaphore):
    # 작업 수행
    print("작업 시작")
    semaphore.release()


if __name__ == "__main__":
    semaphore = Semaphore(0)
    processes = []

    for _ in range(5):  # 예시로 5개의 프로세스를 생성
        p = Process(target=worker, args=(semaphore,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  # 모든 프로세스가 종료될 때까지 기다림

    print("모든 작업 완료")
