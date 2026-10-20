from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'race_docking_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        (os.path.join('share', package_name), ["package.xml"]),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='Search-phase-only mission node for the RACE AUV docking redesign.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'search_mission_node = race_docking_mission.search_mission_node:main',
        ],
    },
)
