from string_art.config_validator import adapt_thread_thickness


def test_adapted_thread_thickness():
    adapted_thread_thickness = adapt_thread_thickness(thread_thickness=0.15, frame_diameter=614.4)
    assert adapt_thread_thickness == 1.5
