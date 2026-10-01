from jetson_app.droplog import DropCounter


def test_drop_counter_logs_first_drop_then_every_n(capsys):
    counter = DropCounter("label", log_every=3)

    counter.add()  # 1: 첫 발생은 항상 출력
    counter.add()  # 2
    counter.add()  # 3: 3의 배수를 처음 넘는 순간 출력
    counter.add()  # 4

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["[label] 누적 1건", "[label] 누적 3건"]
    assert counter.total == 4


def test_drop_counter_add_many_at_once_logs_when_crossing_a_boundary(capsys):
    counter = DropCounter("label", log_every=100)

    counter.add(250)

    assert capsys.readouterr().out.strip() == "[label] 누적 250건"
    assert counter.total == 250
