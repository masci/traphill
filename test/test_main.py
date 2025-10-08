from unittest import mock

from traphill.main import get_trap_area


def test_get_trap_area():
    mocked_cap = mock.MagicMock()
    mocked_cap.get.side_effect = [100, 99]
    area = get_trap_area(mocked_cap)
    assert area.height == 99
    assert area.x1 == 30
    assert area.x2 == 70
    mocked_cap.get.side_effect = [100, 99]
    area = get_trap_area(mocked_cap, 50)
    assert area.x1 == 25
    assert area.x2 == 75
