import xlrd
import xlwt

def write_excel(file_path, sheet_name):
    workbook0 = xlrd.open_workbook(file_path[0], formatting_info=True)
    sheet0 = workbook0.sheet_by_name(sheet_name[0])

    workbook1 = xlrd.open_workbook(file_path[1], formatting_info=True)
    sheet1 = workbook1.sheet_by_name(sheet_name[1])

    workbook2 = xlrd.open_workbook(file_path[2], formatting_info=True)
    sheet2 = workbook2.sheet_by_name(sheet_name[2])


    write_book = xlwt.Workbook(encoding='utf-8')
    write_sheet = write_book.add_sheet(u"Sheet", cell_overwrite_ok = True)

    k = 0

    for i in range(1,sheet2.nrows):
        find = False
        if i % 100==0:
            print(i,'/17772', k)
        for r in range(1,sheet0.nrows):
            if sheet0.cell_value(r, 5) == sheet2.cell_value(i, 0)+'.docx':
                find = True
                if sheet0.cell_value(r, 8) != '/':
                    if abs(int(sheet0.cell_value(r, 8)[:-3])-int(sheet0.cell_value(r, 6)[:-3]))<20:
                        if int(sheet0.cell_value(r, 8)[:-3]) > 95 and int(sheet0.cell_value(r, 6)[:-3]) > 95:
                            write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                            write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                            write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                            write_sheet.write(k, 3, 'A')
                            k = k + 1

                        elif int(sheet0.cell_value(r, 8)[:-3]) < 88 and int(sheet0.cell_value(r, 6)[:-3]) < 88:
                            write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                            write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                            write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                            write_sheet.write(k, 3, 'C')
                            k = k + 1

                        elif int(sheet0.cell_value(r, 8)[:-3]) < 94 and int(sheet0.cell_value(r, 6)[:-3]) < 94 and \
                                    int(sheet0.cell_value(r, 8)[:-3]) > 90 and int(sheet0.cell_value(r, 6)[:-3]) > 90:
                            write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                            write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                            write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                            write_sheet.write(k, 3, 'B')
                            k = k + 1

                        break
        if not find:
            for r in range(1, sheet1.nrows):
                if sheet1.cell_value(r, 5) == sheet2.cell_value(i, 0)+'.docx':
                    if sheet1.cell_value(r, 8) != '/':
                        if abs(int(sheet1.cell_value(r, 8)[:-3]) - int(sheet1.cell_value(r, 6)[:-3])) < 20:
                            if int(sheet1.cell_value(r, 8)[:-3]) > 96 and int(sheet1.cell_value(r, 6)[:-3]) > 96:
                                write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                                write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                                write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                                write_sheet.write(k, 3, 'A')
                                k = k + 1
                            elif int(sheet1.cell_value(r, 8)[:-3]) < 87 and int(sheet1.cell_value(r, 6)[:-3]) < 87:
                                write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                                write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                                write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                                write_sheet.write(k, 3, 'C')
                                k = k + 1
                            elif int(sheet1.cell_value(r, 8)[:-3]) < 94 and int(sheet1.cell_value(r, 6)[:-3]) < 94 and \
                                    int(sheet1.cell_value(r, 8)[:-3]) > 90 and int(sheet1.cell_value(r, 6)[:-3]) > 90:
                                write_sheet.write(k, 0, sheet2.cell_value(i, 0))  # id
                                write_sheet.write(k, 1, sheet2.cell_value(i, 1))  # tf
                                write_sheet.write(k, 2, sheet2.cell_value(i, 2))  # if
                                write_sheet.write(k, 3, 'B')
                                k = k + 1
                            break





    write_book.save('example3.xls')


