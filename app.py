from flask import Flask, render_template, request, jsonify
from model import analyze_new_school
from teacher_allocate import find_teacher_need, create_df
import numpy as np
app = Flask(__name__)

standardized_cluster_centroid_ssa = np.array([[-0.16926386, -0.16668369, -0.00475765, -0.16603637, -0.00095075,
        -0.00125603,  0.00141692,  0.08221864, -0.19284485,  0.10555196,
        -0.17144971,  0.10004174, -0.17349677, -0.00199627,  0.01000215,
         0.0176234 , -0.00559707, -0.00746518,  0.03358864,  0.00855581,
         0.00441448, -0.09018103, -0.03717806, -0.00227185,  0.00238509,
        -0.00177652, -0.02553753, -0.0128966 , -0.01658209,  0.00078473,
        -0.017146  , -0.01719746, -0.00187928, -0.00509202, -0.00407335,
        -0.00465518, -0.01172053, -0.01166889, -0.01191348, -0.00613735,
        -0.00623581, -0.00381185, -0.00485801,  0.00182558, -0.00794176,
         0.0121599 ,  0.02213284, -0.01432962,  0.00560709,  0.00097479,
        -0.02218254, -0.03565175,  0.04261397,  0.03606982, -0.04476473,
         0.00579473, -0.00518705,  0.0041457 ,  0.00235832, -0.0014901 ,
         0.00105333,  0.00105333,  0.06297   ,  0.00701391, -0.16969329,
        -0.13706743, -0.14412739,  0.20102555,  0.00078117,  0.00196143,
        -0.00561534, -0.03361821,  0.00625821,  0.01401936,  0.00163548,
         0.00210867, -0.0094108 ]])


columns = [
    'School Lat', 'School Long', 'Rural/Urban', 'Student-Teacher Ratio',
    'Total Students', 'Total Teachers',
    'Mathematics Students', 'Mathematics Teachers', 'Nutrition Students', 'Nutrition Teachers',
    'Biology Students', 'Biology Teachers', 'Business Studies Students', 'Business Studies Teachers',
    'Geography Students', 'Geography Teachers', 'Science Students', 'Science Teachers',
    'Statistics Students', 'Statistics Teachers', 'Physical Education Students', 'Physical Education Teachers',
    'Computer Studies Students', 'Computer Studies Teachers', 'Computer Application Students', 'Computer Application Teachers',
    'Work Education Students', 'Work Education Teachers', 'Accountancy Students', 'Accountancy Teachers',
    'Economics Students', 'Economics Teachers', 'Electronics Students', 'Electronics Teachers',
    'Psychology Students', 'Psychology Teachers', 'Political Science Students', 'Political Science Teachers',
    'History Students', 'History Teachers', 'Bengali Students', 'Bengali Teachers',
    'Social Science Students', 'Social Science Teachers', 'Cost and Taxation Students', 'Cost and Taxation Teachers',
    'Music Students', 'Music Teachers', 'Chemistry Students', 'Chemistry Teachers',
    'Sociology Students', 'Sociology Teachers', 'Fine Arts Students', 'Fine Arts Teachers',
    'Hindi Students', 'Hindi Teachers', 'Education Students', 'Education Teachers',
    'Sanskrit Students', 'Sanskrit Teachers', 'Philosophy Students', 'Philosophy Teachers',
    'Environmental Studies (EVS) Students', 'Environmental Studies (EVS) Teachers', 'English Students', 'English Teachers',
    'Art Education Students', 'Art Education Teachers', 'Physics Students', 'Physics Teachers',
    'Computer Science Students', 'Computer Science Teachers',
    'Mathematics Demand', 'Mathematics Teachers Needed', 'Nutrition Demand', 'Nutrition Teachers Needed',
    'Biology Demand', 'Biology Teachers Needed', 'Business Studies Demand', 'Business Studies Teachers Needed',
    'Geography Demand', 'Geography Teachers Needed', 'Science Demand', 'Science Teachers Needed',
    'Statistics Demand', 'Statistics Teachers Needed', 'Physical Education Demand', 'Physical Education Teachers Needed',
    'Computer Studies Demand', 'Computer Studies Teachers Needed', 'Computer Application Demand', 'Computer Application Teachers Needed',
    'Work Education Demand', 'Work Education Teachers Needed', 'Accountancy Demand', 'Accountancy Teachers Needed',
    'Economics Demand', 'Economics Teachers Needed', 'Electronics Demand', 'Electronics Teachers Needed',
    'Psychology Demand', 'Psychology Teachers Needed', 'Political Science Demand', 'Political Science Teachers Needed',
    'History Demand', 'History Teachers Needed', 'Bengali Demand', 'Bengali Teachers Needed',
    'Social Science Demand', 'Social Science Teachers Needed', 'Cost and Taxation Demand', 'Cost and Taxation Teachers Needed',
    'Music Demand', 'Music Teachers Needed', 'Chemistry Demand', 'Chemistry Teachers Needed',
    'Sociology Demand', 'Sociology Teachers Needed', 'Fine Arts Demand', 'Fine Arts Teachers Needed',
    'Hindi Demand', 'Hindi Teachers Needed', 'Education Demand', 'Education Teachers Needed',
    'Sanskrit Demand', 'Sanskrit Teachers Needed', 'Philosophy Demand', 'Philosophy Teachers Needed',
    'Environmental Studies (EVS) Demand', 'Environmental Studies (EVS) Teachers Needed', 'English Demand', 'English Teachers Needed',
    'Art Education Demand', 'Art Education Teachers Needed', 'Physics Demand', 'Physics Teachers Needed',
    'Computer Science Demand', 'Computer Science Teachers Needed'
]
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data and convert it into a dictionary
        form_data = {
            'library_availability': request.form.get('library_availability'),
            'playground_available': request.form.get('playground_available'),
            'medical_checkups': request.form.get('medical_checkups'),
            'availability_ramps': request.form.get('availability_ramps'),
            'availability_of_handrails': request.form.get('availability_of_handrails'),
            'handwash_near_toilet': request.form.get('handwash_near_toilet'),
            'drinking_water_available': request.form.get('drinking_water_available'),
            'drinking_water_functional': request.form.get('drinking_water_functional'),
            'rain_water_harvesting': request.form.get('rain_water_harvesting'),
            'handwash_facility_for_meal': request.form.get('handwash_facility_for_meal'),
            'electricity_availability': request.form.get('electricity_availability'),
            'approachable_road': request.form.get('approachable_road'),
            'anganwadi_premises': request.form.get('anganwadi_premises'),
            'special_training': request.form.get('special_training'),
            'smc_exists': request.form.get('smc_exists'),
            'separate_room_for_hm': request.form.get('separate_room_for_hm'),
            'anganwadi_worker': request.form.get('anganwadi_worker'),
            'cce_pr': request.form.get('cce_pr'),
            'cce_up': request.form.get('cce_up'),
            'smdc': request.form.get('smdc_constituted'),
            'solar': request.form.get('solar_panel'),
            'transport_pr': request.form.get('transport_pr'),
            'building_status': request.form.get('building_status'),
            'lowclass': request.form.get('lowclass'),
            'highclass': request.form.get('highclass'),
            'anganwadi_boys': request.form.get('anganwadi_boys'),
            'anganwadi_girls': request.form.get('anganwadi_girls'),
            'instr_days_pr': request.form.get('instr_days_pr'),
            'instr_days_up': request.form.get('instr_days_up'),
            'avg_school_hrs_student_pr': request.form.get('avg_school_hrs_student_pr'),
            'avg_school_hrs_student_up': request.form.get('avg_school_hrs_student_up'),
            'avg_school_hrs_teacher_pr': request.form.get('avg_school_hrs_teacher_pr'),
            'avg_school_hrs_teacher_up': request.form.get('avg_school_hrs_teacher_up'),
            'acad_inspections': request.form.get('acad_inspections'),
            'crc_coordinator': request.form.get('crc_coordinator'),
            'block_level_officers': request.form.get('block_level_officers'),
            'district_officers': request.form.get('district_officers'),
            'free_text_books_pr': request.form.get('free_text_books_pr'),
            'free_uniform_pr': request.form.get('free_uniform_pr'),
            'free_text_books_up': request.form.get('free_text_books_up'),
            'free_uniform_up': request.form.get('free_uniform_up'),
            'no_building_blocks': request.form.get('no_building_blocks'),
            'pucca_building_blocks': request.form.get('pucca_building_blocks'),
            'boundary_wall': request.form.get('boundary_wall'),
            'total_class_rooms': request.form.get('total_class_rooms'),
            'other_rooms': request.form.get('other_rooms'),
            'classrooms_in_good_condition': request.form.get('classrooms_in_good_condition'),
            'classrooms_needs_minor_repair': request.form.get('classrooms_needs_minor_repair'),
            'classrooms_needs_major_repair': request.form.get('classrooms_needs_major_repair'),
            'total_boys_toilet': request.form.get('total_boys_toilet'),
            'total_boys_func_toilet': request.form.get('total_boys_func_toilet'),
            'total_girls_toilet': request.form.get('total_girls_toilet'),
            'total_girls_func_toilet': request.form.get('total_girls_func_toilet'),
            'func_boys_cwsn_friendly': request.form.get('func_boys_cwsn_friendly'),
            'func_girls_cwsn_friendly': request.form.get('func_girls_cwsn_friendly'),
            'urinal_boys': request.form.get('urinal_boys'),
            'urinla_girls': request.form.get('urinla_girls'),
            'school_type': request.form.get('school_type'),
        }
        
        # Return the dictionary as a string (for demonstration)
        for key in form_data:
            if form_data[key] is None:
                form_data[key] = 0
            if form_data[key] == '':
                form_data[key] = 0
        
        recommendations, standardized_percentage = analyze_new_school(new_school_data=form_data)
        return jsonify({
            'standardization percentage': standardized_percentage,
            'recommendations': recommendations
        })

    return render_template('form.html')

@app.route('/allocate', methods='POST')
def allocate_teacher(test_df):
    synthetic_data = create_df(test_df)
    synthetic_data = synthetic_data[columns]
    Y_pred_orig = find_teacher_need(synthetic_data)
    return Y_pred_orig

if __name__ == '__main__':
    app.run(debug=True)
