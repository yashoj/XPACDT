#!/bin/bash

source ~/code/XPACDT/.XPACDT

xpacdt.py -i full_A-1-1.in &> full_A-1-1.out
xpacdt.py -i full_A-1-4.in &> full_A-1-4.out
xpacdt.py -i full_A-8-1.in &> full_A-8-1.out
xpacdt.py -i full_A-8-32.in &> full_A-8-32.out

xpacdt.py -i full_Q-1-1.in &> full_Q-1-1.out
xpacdt.py -i full_Q-1-4.in &> full_Q-1-4.out
xpacdt.py -i full_Q-8-1.in &> full_Q-8-1.out
xpacdt.py -i full_Q-8-32.in &> full_Q-8-32.out

xpacdt.py -i full_HO-1-1.in &> full_HO-1-1.out
xpacdt.py -i full_HO-1-4.in &> full_HO-1-4.out
xpacdt.py -i full_HO-8-1.in &> full_HO-8-1.out
xpacdt.py -i full_HO-8-32.in &> full_HO-8-32.out

